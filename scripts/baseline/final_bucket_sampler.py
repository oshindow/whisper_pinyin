"""Shuffle final/length buckets, then shard intact microbatches across DDP ranks."""
import math
import random
from collections import defaultdict

import torch.distributed as dist
from torch.utils.data import Sampler


class FinalBucketSampler(Sampler):
    def __init__(self, rows, batch_size, world_size=1, seed=42, pool_batches=32):
        self.batch_size, self.world_size = batch_size, world_size
        self.seed, self.epoch = seed, 0
        self.pool_size = batch_size * pool_batches
        self.tones, self.speakers, self.accents, self.durations = [], [], [], []
        for row in rows:
            tones = defaultdict(set)
            for phone in row['actual_phones']:
                if phone and phone[-1:] in '12345':
                    tones[phone[:-1]].add(phone[-1])
            self.tones.append(dict(tones))
            speaker = row.get('speaker_id')
            self.speakers.append((str(row.get('source', '')), str(speaker))
                                 if speaker is not None and str(speaker) else None)
            accent = row.get('accent_id')
            self.accents.append(str(accent) if accent is not None and str(accent) else None)
            duration = row.get('duration')
            if duration is None or not math.isfinite(float(duration)) or float(duration) <= 0:
                raise ValueError('Final bucket sampling requires positive duration in the prepared manifest')
            self.durations.append(float(duration))

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        # Drop at most world_size * batch_size - 1 samples; no repeated examples.
        return len(self.tones) // (self.batch_size * self.world_size) * self.batch_size

    def batches(self):
        rng = random.Random(self.seed + self.epoch)
        buckets = defaultdict(list)
        for i, tones in enumerate(self.tones):
            final = rng.choice(sorted(tones)) if tones else None
            buckets[final].append(i)
        batches, leftovers = [], []
        for final, indices in buckets.items():
            # Random boundary offset + local shuffle avoid fixed length neighbours.
            rng.shuffle(indices)
            indices.sort(key=lambda i: self.durations[i])
            offset = rng.randrange(self.pool_size)
            pools = [indices[:offset]] if offset else []
            pools += [indices[i:i + self.pool_size] for i in range(offset, len(indices), self.pool_size)]
            for pool in pools:
                rng.shuffle(pool)
                while len(pool) >= self.batch_size:
                    anchor = pool.pop()
                    batch = [anchor]
                    tone = rng.choice(sorted(self.tones[anchor][final])) if final else None
                    positive_added = negative_added = False
                    while len(batch) < self.batch_size:
                        def priority(j):
                            other = self.tones[j].get(final, set())
                            positive = (tone in other and self.speakers[anchor] is not None
                                        and self.speakers[j] is not None
                                        and self.speakers[j] != self.speakers[anchor])
                            negative = bool(other - {tone}) if tone else False
                            cross_accent = (self.accents[anchor] is not None and self.accents[j] is not None
                                            and self.accents[anchor] != self.accents[j])
                            if not positive_added and positive:
                                return 4 + int(cross_accent)
                            if not negative_added and negative:
                                return 3
                            return int(positive) + int(negative)
                        # Pool is shuffled, so ties are resolved randomly.
                        j = max(pool, key=priority)
                        pool.remove(j)
                        other = self.tones[j].get(final, set())
                        positive_added |= (tone in other and self.speakers[anchor] is not None
                                           and self.speakers[j] is not None
                                           and self.speakers[j] != self.speakers[anchor])
                        negative_added |= bool(other - {tone}) if tone else False
                        batch.append(j)
                    batches.append(batch)
                leftovers.extend(pool)
        rng.shuffle(leftovers)
        batches.extend(leftovers[i:i + self.batch_size]
                       for i in range(0, len(leftovers) - self.batch_size + 1, self.batch_size))
        rng.shuffle(batches)
        # Equal steps on every rank; rotate discarded tail via the epoch seed.
        return batches[:len(batches) // self.world_size * self.world_size]

    def __iter__(self):
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        return iter([i for batch in self.batches()[rank::self.world_size] for i in batch])
