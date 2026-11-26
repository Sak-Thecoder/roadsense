# RoadSense baseline status

> **Audit record from 2026-11-15.** It describes the data before training. Since then the StreetSurfaceVis mapping and split were approved (2026-11-16) and the quality_v3 models were trained on them. See [docs/DATASET_UTILIZATION.md](../../../docs/DATASET_UTILIZATION.md) for how the data is used now.

## Labels and permissions

| Source | Images | Original or proposed quality counts | Verified RoadSense labels | Training-ready now |
|---|---:|---|---|---:|
| StreetSurfaceVis | 9,122 | Proposed Good 4,908; Moderate 2,610; Poor 1,604 | 0, awaiting mapping approval | 0 |
| Cambodian | 100 | Existing decisions confirmed by owner | Good 39; Moderate 40; Poor 20; Uncertain 1 | 0 |
| RDD_SPLIT | 38,385 | Damage boxes only; mapping not supplied | 0 | 0 |

The owner confirmed the 95 recorded values previously marked unreviewed and the earlier five Poor values remain intact. Self-collected image labels: Good 15, Moderate 30, Poor 5; ownership and training authorization confirmed for all 50. Online-image labels: Good 24, Moderate 10, Poor 15, Uncertain 1; training permission remains unconfirmed. Review date and reviewer name remain blank, not fabricated. All Cambodian capture groups remain unverified; CAM022/CAM023 share an exact duplicate. Thus verified label counts are not training eligibility counts.

StreetSurfaceVis proposals by source split: train Good 4,325 / Moderate 2,503 / Poor 1,518; test Good 583 / Moderate 107 / Poor 86. The source authors' original five-quality labels and all CSV fields are retained. The mapping is a plausible ordinal aggregation, not validated as equivalent to RoadSense visual criteria. The 99-image review gallery covers 34 rating/surface/split strata; its balanced design is useful for review but is not a prevalence-representative random sample. Excellent and good vs intermediate are ambiguous near the class boundary; bad and very_bad have different severity, and unpaved paths and sidewalks may fall outside the target use. Approve a rubric and decide scope after reviewing the gallery and guide.

## RDD pilot and mapping

The existing `work/full/pilot.csv` has 450 diverse RDD images (397 nonempty, 53 empty annotations) selected across country prefixes and numeric classes. It has not acquired quality labels. Recursive inspection of the supplied RDD tree and the corresponding paths in `dataset.zip` found no authoritative mapping, XML, YAML, or class definition; preserve numeric IDs. The earlier audit remains unchanged.

## Steps before training

1. Review gallery and approve or revise the StreetSurfaceVis mapping and surface scope. Assess terms, attribution and intended model/distribution use for CC BY-SA 4.0.
2. Supply actual reviewer name and original review date where known; retain blank where unknown. Verify capture sessions/road segments for the Cambodian set, and review duplicate candidates. Clear online licenses individually or exclude those images.
3. Investigate direct cross-split and cross-source matches; preserve the authors' geographic test set. Obtain sequence IDs or conservative location/session groups, then construct validation from training groups and inspect class balance.
4. Obtain RDD class mapping and license; use the 450-image pilot only after manual quality labels. Reserve Cambodian evaluation images only once independent groups and locations are verified.
5. Generate a new versioned training-ready manifest after the evidence is recorded. Apply future augmentation only to the training partition. No CNN training has been run.
