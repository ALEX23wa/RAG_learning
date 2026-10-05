# CPU model benchmark v4

The Qwen embedding CPU probe exposed a compute-precision mismatch: the local
checkpoint selected BF16 parameters by default, while SentenceTransformer
returned a float32 NumPy array after normalization had already happened in BF16.
On four generic, non-evaluation passages all outputs were finite, but the largest
L2 norm error was 0.00159013. Converting that same model to FP32 gave a maximum
norm error of 0 and passed the existing tolerance. The probe process exited;
peak worker RSS was 3.115 GiB. This was a numerical adapter check, not a quality
comparison. Its first/default and later/FP32 timings have different warm-up state
and do not establish a model speed improvement.

## Precision and compatibility

New execution writes `schema: model-benchmark-v4`. CPU embedding construction
passes `model_kwargs={"torch_dtype": torch.float32}` and verifies every parameter
before creating or reading the vector cache. The following report fields are
recorded:

- `settings.embedding_parameter_dtype`: `torch.float32` for embedding runs.
- `embedding_precision.requested_parameter_dtype`: `torch.float32`.
- `embedding_precision.actual_parameter_dtypes`: exactly `["torch.float32"]`.
- `embedding_precision.validation`: `all_parameters_checked`.

The full precision record is included in the document cache contract and v4
query-commit identity. Resume rejects a different recorded actual precision.
Vector shape, finiteness and L2 norm checks retain the absolute tolerance
`2e-4`; output casting does not bypass them. Unknown, mixed or lower-precision
parameters fail before inference or cache creation.

Complete, valid v3 and v4 dense reports can both supply a fixed reranking pool.
The v3 query-commit calculation is preserved. A v4 executable cannot resume a
v3 report or vector cache; suite, model bytes, source code, package versions and
settings must still match. Old reports, caches and source snapshots are not
migrated or rewritten. A v3 NumPy dtype does not establish its parameter dtype;
unrecorded computation precision must be reported as unknown.

New reranker runs also request and verify FP32 parameters. BGE CrossEncoder now
receives the explicit dtype at construction; Qwen already requested FP32.
`settings.reranker_parameter_dtype` records the requirement. The top-level
`rerank_precision` contains `requested_parameter_dtype`,
`actual_parameter_dtypes`, and `validation: all_parameters_checked`. New query
commits bind this field when it is present, and resume checks the actual record.
The presence-gated addition preserves existing v3/v4 embedding commits and old
reranking commits. Old reranking precision remains unknown; a new declared
precision without actual validation is rejected.

For each candidate window, the shared reranker measures the complete model
input without truncation before calling the model: BGE query/document pairs
include special tokens, while Qwen includes its instruction and chat template.
The effective limit is the smaller of the configured limit and the adapter's
configured maximum. An over-budget pair fails before inference; it is not
silently shortened. Valid traces include `rerank_max_pair_tokens`,
`rerank_model_max_tokens`, `rerank_pair_budget_verified: true`, and
`silent_truncations: 0`. An adapter without a tokenizer cannot claim verified
lengths or zero truncations. Document and question windows still preserve their
tails; per-candidate scoring still takes the maximum window score.

## Resource sampling and progress

`resources.sampling_protocol` and `settings.resource_sampling_protocol` are
`target-pid-rss-v2`. The monitor samples worker, recursive children and discovered
Qdrant process RSS. It reads memory only for those PIDs, counts overlapping child
and database PIDs once, and uses name-only database discovery every five seconds.
On Windows, protected executable names unavailable through psutil are resolved
using a read-only Toolhelp process-name snapshot; unknown live names or a failed
fallback are not silently ignored.
The background sampling interval remains 0.2 seconds; synchronous boundary
checks use the same discovery state. This is periodic observation, not a claim
to observe every instantaneous allocation or a newly started database before its
next discovery. Exited processes can disappear normally; live target access or
sampling failures yield unavailable resource status and fail closed. A recorded
RSS peak remains latched after memory falls.

Tokenization checkpoints report document counts every 128 documents and at both
ends. Index reuse checkpoints report source validation, subset recomputation and
import persistence separately. Vector markers remain atomic, checksum verified
and resumable. `--batch-size` still controls model inference and defaults to four.
The optional `--import-batch-size` controls only already-verified index vector
storage, defaults to 128, and accepts 1–1024. All previous embed options remain
available; no new argument is required.

## Interpretation across experiment epochs

An earlier read-only diagnosis found 25 import batches committed in 20.03 seconds
while full-system RSS enumeration cost approximately 0.583 seconds per scan.
The old implementation scanned all 335 processes for each small import batch as
well as in its background thread. The worker was making progress; tokenization
was not the bottleneck. v4 changes this measurement and persistence overhead.

Compare quality on the same frozen suite with explicit precision and code-epoch
disclosures. Do not describe v3/v4 wall-time numbers as an otherwise identical
performance comparison. Index verification/import time includes checksums,
resource monitoring, disk IO and previously computed vectors: **it is never model
encoding throughput**. Full application latency and service/database memory
acceptance still require their separate frozen evaluation.

Embedding FP32 changes apply only to the offline benchmark; applying that
embedding requirement to the application service remains a separate
production-readiness task. Reranker FP32 and pair-length checks are implemented
in the shared retrieval adapter used by both the benchmark and application.
