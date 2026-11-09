-- Per-profile projection from one exported nsys SQLite into fact-table shape.
--
-- build.py ATTACHes a single <run>/<label>_rank<n>.sqlite as `s`, sets the run,
-- label and rank variables, then runs this file. Each view below is one
-- profile's contribution to one fact table; build.py concatenates them across
-- every profile and writes the parquet.
--
-- nsys only writes a table when it has something to put in it, so anything a
-- profile may lack is read through a src_<table> view, which build.py points at
-- the real table or at an empty one. Tables read as s.<table> always exist.
--
-- Nothing is aggregated here. Raw ns timestamps are kept alongside the derived
-- ms/us columns so a query can always drop back to source resolution.
--
-- Only the tables worth keeping are projected. In particular OSRT_API is
-- ignored: it is ~403k rows per profile (90% of them `accept`) and accounts for
-- essentially the entire size of the .sqlite, while nothing downstream reads it.
--
-- CUPTI_ACTIVITY_KIND_RUNTIME is joined but not projected. At ~11k rows per
-- profile it would be affordable, but the only thing downstream wants from it is
-- when and on which thread a given kernel or copy was launched -- so that is
-- denormalised onto v_kernels/v_memops as launch_ns/launch_tid and the table
-- itself stays out. ATTACH a raw .sqlite if you need the API calls themselves.

-- One row, cross-joined into every other view so the identifying columns and
-- the epoch offset are defined in exactly one place.
--
-- rank is resolved by build.py before this file runs: MPI_RANKS is preferred, as
-- the record of what the process actually was, and is cross-checked against the
-- filename when both exist.
CREATE OR REPLACE TEMP VIEW v_profiles AS
SELECT
    getvariable('run')                                        AS run,
    getvariable('label')                                      AS label,
    CAST(getvariable('rank') AS BIGINT)                       AS rank,
    -- Session start as absolute UTC ns. Every other timestamp in an nsys
    -- profile is per-process and relative to this, so adding it is what makes
    -- one rank's timeline comparable with another's.
    (SELECT utcEpochNs FROM s.TARGET_INFO_SESSION_START_TIME) AS utc_epoch_ns,
    (SELECT utcTime    FROM s.TARGET_INFO_SESSION_START_TIME) AS session_utc,
    -- The node the rank ran on, e.g. jpbo-103-23.jupiter.internal.
    (SELECT value FROM src_target_info_system_env
      WHERE name = 'Hostname' LIMIT 1)                        AS hostname,
    (SELECT name       FROM src_target_info_gpu LIMIT 1)        AS gpu_name,
    (SELECT count(*)   FROM src_target_info_gpu)                AS gpu_count,
    (SELECT value FROM s.META_DATA_EXPORT
      WHERE name = 'EXPORT_PRODUCT_VERSION')                  AS nsys_version,
    -- Asserted equal across all profiles by build.py: the export schema shifts
    -- between nsys releases, and silently mixing two of them in one fact table
    -- is how a longitudinal comparison quietly stops meaning anything.
    (SELECT value FROM s.META_DATA_EXPORT
      WHERE name = 'EXPORT_SCHEMA_VERSION')                   AS schema_version;

-- Every completed push/pop range. Deliberately unfiltered: the rep/warmup
-- bucketing and the stage-name matching are query-time concerns, not
-- extraction-time ones.
CREATE OR REPLACE TEMP VIEW v_nvtx AS
SELECT p.run, p.label, p.rank,
       -- Domain-qualified names arrive with a leading ':'.
       ltrim(coalesce(e.text, si.value), ':')  AS name,
       e.start                                 AS start_ns,
       e."end"                                 AS end_ns,
       (e."end" - e.start) / 1e6               AS dur_ms,
       p.utc_epoch_ns + e.start                AS abs_start_ns,
       e.globalTid % 16777216                  AS tid
FROM src_nvtx_events e
LEFT JOIN s.StringIds si ON si.id = e.textId
CROSS JOIN v_profiles p
WHERE e."end" IS NOT NULL
  AND coalesce(e.text, si.value) IS NOT NULL;

-- launch_ns/launch_tid are the host-side CUDA API call that submitted this
-- kernel, recovered through correlationId. They are what makes a kernel
-- attributable to an NVTX stage: the kernel itself runs asynchronously and may
-- outlive the range it belongs to, but the launch that submitted it happened on
-- the calling thread, inside the range that was open at the time.
--
-- LEFT JOIN, not INNER: a kernel replayed from a CUDA graph correlates to the
-- graph launch rather than to its own submission, and would otherwise vanish.
-- For code that doesn't use graphs, a NULL launch_ns is worth looking at.
CREATE OR REPLACE TEMP VIEW v_kernels AS
SELECT p.run, p.label, p.rank,
       short.value                             AS name,
       demangled.value                         AS demangled,
       k.start                                 AS start_ns,
       k."end"                                 AS end_ns,
       (k."end" - k.start) / 1e3               AS dur_us,
       p.utc_epoch_ns + k.start                AS abs_start_ns,
       r.start                                 AS launch_ns,
       r.globalTid % 16777216                  AS launch_tid,
       k.deviceId                              AS device_id,
       k.streamId                              AS stream_id,
       k.correlationId                         AS correlation_id,
       k.graphNodeId                           AS graph_node_id,
       k.gridX, k.gridY, k.gridZ,
       k.blockX, k.blockY, k.blockZ,
       k.registersPerThread                    AS registers_per_thread,
       k.staticSharedMemory                    AS static_smem,
       k.dynamicSharedMemory                   AS dynamic_smem
FROM src_cupti_activity_kind_kernel k
LEFT JOIN s.StringIds short     ON short.id     = k.shortName
LEFT JOIN s.StringIds demangled ON demangled.id = k.demangledName
LEFT JOIN src_cupti_activity_kind_runtime r ON r.correlationId = k.correlationId
CROSS JOIN v_profiles p;

-- Copies and sets in one table, discriminated by `kind` -- the same shape as
-- v_mpi below. They are folded together because the question they answer is
-- always the same one: how much non-kernel work the GPU did inside a stage.
-- Splitting them across two fact tables just means every such query starts
-- with a UNION ALL.
--
-- `op` is the copy direction (DtoH, HtoD, ...) for a copy and NULL for a set,
-- which has no direction. MEMSET has no bytes-per-second worth reporting
-- either, but it does have bytes, so the column is shared.
CREATE OR REPLACE TEMP VIEW v_memops AS
SELECT p.run, p.label, p.rank,
       'memcpy'                                AS kind,
       op.label                                AS op,
       m.start                                 AS start_ns,
       m."end"                                 AS end_ns,
       (m."end" - m.start) / 1e3               AS dur_us,
       p.utc_epoch_ns + m.start                AS abs_start_ns,
       r.start                                 AS launch_ns,
       r.globalTid % 16777216                  AS launch_tid,
       m.bytes,
       -- bytes/ns is exactly GB/s (1e9 bytes / 1e9 ns), no scaling needed.
       m.bytes / nullif(m."end" - m.start, 0)  AS gb_per_s,
       m.deviceId                              AS device_id,
       m.streamId                              AS stream_id
FROM src_cupti_activity_kind_memcpy m
LEFT JOIN src_enum_cuda_memcpy_oper op ON op.id = m.copyKind
LEFT JOIN src_cupti_activity_kind_runtime r ON r.correlationId = m.correlationId
CROSS JOIN v_profiles p
UNION ALL
SELECT p.run, p.label, p.rank,
       'memset'                                AS kind,
       NULL                                    AS op,
       z.start                                 AS start_ns,
       z."end"                                 AS end_ns,
       (z."end" - z.start) / 1e3               AS dur_us,
       p.utc_epoch_ns + z.start                AS abs_start_ns,
       r.start                                 AS launch_ns,
       r.globalTid % 16777216                  AS launch_tid,
       z.bytes,
       z.bytes / nullif(z."end" - z.start, 0)  AS gb_per_s,
       z.deviceId                              AS device_id,
       z.streamId                              AS stream_id
FROM src_cupti_activity_kind_memset z
LEFT JOIN src_cupti_activity_kind_runtime r ON r.correlationId = z.correlationId
CROSS JOIN v_profiles p;

-- Collectives and point-to-point in one table, discriminated by `kind`.
-- MPI_OTHER_EVENTS has no size/root columns, hence the nulls.
CREATE OR REPLACE TEMP VIEW v_mpi AS
SELECT p.run, p.label, p.rank,
       'collective'                            AS kind,
       si.value                                AS name,
       c.start                                 AS start_ns,
       c."end"                                 AS end_ns,
       (c."end" - c.start) / 1e6               AS dur_ms,
       p.utc_epoch_ns + c.start                AS abs_start_ns,
       c.size, c.recvSize                      AS recv_size,
       c.rootRank                              AS root_rank
FROM src_mpi_collectives_events c
LEFT JOIN s.StringIds si ON si.id = c.textId
CROSS JOIN v_profiles p
UNION ALL
SELECT p.run, p.label, p.rank,
       'other'                                 AS kind,
       si.value                                AS name,
       o.start                                 AS start_ns,
       o."end"                                 AS end_ns,
       (o."end" - o.start) / 1e6               AS dur_ms,
       p.utc_epoch_ns + o.start                AS abs_start_ns,
       NULL, NULL, NULL
FROM src_mpi_other_events o
LEFT JOIN s.StringIds si ON si.id = o.textId
CROSS JOIN v_profiles p;

-- One row per GPU metrics sample, one column per metric. nsys stores these
-- long (one row per metric per sample, ~3.8M rows a profile at the default
-- 10 kHz); pivoting them wide makes it ~120k rows a profile and lets a query
-- read "the clock while this kernel ran" without a self-join per metric.
--
-- Reads gpu_metrics_src, not s.GPU_METRICS: build.py points that view at the
-- profile's metrics, or at an empty one for profiles taken without
-- --gpu-metrics-devices, which have no GPU_METRICS table at all.
--
-- Dropped: the Vertex/Tess/Geometry and Pixel warp metrics, which are graphics
-- pipeline counters and identically 0 here; and the three "[Avg]" warp
-- columns, which are per-sample warp sums -- the "[Avg Warps per Cycle]" column
-- of the same metric without the sample-length scale.
--
-- No power or temperature: Hopper's GPU metrics set has none. Those come from
-- NVML/DCGM, not from nsys --gpu-metrics-devices.
--
-- device_id is the CUDA ordinal the samples are for. nsys does not record it as
-- a column; it is encoded as the event type's offset from its source, which has
-- been seen to match CUPTI_ACTIVITY_KIND_KERNEL.deviceId on Grace Hopper.
--
-- Samples run from before session start (negative ts_ns) to process exit, so
-- most rows are an idle GPU: interval-join against nvtx/kernels for the window
-- you care about.
CREATE OR REPLACE TEMP VIEW v_gpu_metrics AS
SELECT p.run, p.label, p.rank,
       m.device_id,
       m.ts_ns,
       p.utc_epoch_ns + m.ts_ns                AS abs_ts_ns,
       -- Labelled [MHz] by nsys, but the values are Hz (1.98e9 at boost).
       max(m.value) FILTER (WHERE m.metric = 'GPC Clock Frequency [MHz]') / 1e6
                                               AS gpc_clock_mhz,
       max(m.value) FILTER (WHERE m.metric = 'SYS Clock Frequency [MHz]') / 1e6
                                               AS sys_clock_mhz,
       max(m.value) FILTER (WHERE m.metric = 'GR Active [Throughput %]')
                                               AS gr_active_pct,
       max(m.value) FILTER (WHERE m.metric = 'SMs Active [Throughput %]')
                                               AS sms_active_pct,
       max(m.value) FILTER (WHERE m.metric = 'SM Issue [Throughput %]')
                                               AS sm_issue_pct,
       max(m.value) FILTER (WHERE m.metric = 'Tensor Active [Throughput %]')
                                               AS tensor_active_pct,
       max(m.value) FILTER (WHERE m.metric = 'Compute Warps in Flight [Throughput %]')
                                               AS compute_warps_pct,
       max(m.value) FILTER (WHERE m.metric = 'Compute Warps in Flight [Avg Warps per Cycle]')
                                               AS compute_warps_per_cycle,
       max(m.value) FILTER (WHERE m.metric = 'Unallocated Warps in Active SMs [Throughput %]')
                                               AS unallocated_warps_pct,
       max(m.value) FILTER (WHERE m.metric = 'Unallocated Warps in Active SMs [Avg Warps per Cycle]')
                                               AS unallocated_warps_per_cycle,
       max(m.value) FILTER (WHERE m.metric = 'DRAM Read Bandwidth [Throughput %]')
                                               AS dram_read_pct,
       max(m.value) FILTER (WHERE m.metric = 'DRAM Write Bandwidth [Throughput %]')
                                               AS dram_write_pct,
       max(m.value) FILTER (WHERE m.metric = 'NVLink RX Requests User Data [Throughput %]')
                                               AS nvlink_rx_req_user_pct,
       max(m.value) FILTER (WHERE m.metric = 'NVLink RX Requests Protocol Data [Throughput %]')
                                               AS nvlink_rx_req_proto_pct,
       max(m.value) FILTER (WHERE m.metric = 'NVLink RX Responses User Data [Throughput %]')
                                               AS nvlink_rx_resp_user_pct,
       max(m.value) FILTER (WHERE m.metric = 'NVLink RX Responses Protocol Data [Throughput %]')
                                               AS nvlink_rx_resp_proto_pct,
       max(m.value) FILTER (WHERE m.metric = 'NVLink TX Requests User Data [Throughput %]')
                                               AS nvlink_tx_req_user_pct,
       max(m.value) FILTER (WHERE m.metric = 'NVLink TX Requests Protocol Data [Throughput %]')
                                               AS nvlink_tx_req_proto_pct,
       max(m.value) FILTER (WHERE m.metric = 'NVLink TX Responses User Data [Throughput %]')
                                               AS nvlink_tx_resp_user_pct,
       max(m.value) FILTER (WHERE m.metric = 'NVLink TX Responses Protocol Data [Throughput %]')
                                               AS nvlink_tx_resp_proto_pct,
       -- CTC is the Grace-Hopper NVLink-C2C link to the CPU.
       max(m.value) FILTER (WHERE m.metric = 'CTC RX Throughput [Throughput %]')
                                               AS c2c_rx_pct,
       max(m.value) FILTER (WHERE m.metric = 'CTC TX Throughput [Throughput %]')
                                               AS c2c_tx_pct,
       max(m.value) FILTER (WHERE m.metric = 'PCIe RX Throughput [Throughput %]')
                                               AS pcie_rx_pct,
       max(m.value) FILTER (WHERE m.metric = 'PCIe TX Throughput [Throughput %]')
                                               AS pcie_tx_pct
FROM gpu_metrics_src m
CROSS JOIN v_profiles p
GROUP BY ALL;
