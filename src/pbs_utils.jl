# GOOD APPROACH TO CHECKING CGROUP MEM USAGE
# memory info for the cgroup of a job is in
#   /sys/fs/cgroup/memory/pbspro.service/jobid/[JOBID]
# where the last string is ENV["PBS_JOBID"], like 4272839.pbsha.ib.sockeye
# In particular, the files
#   memory.limit_in_bytes
#   memory.usage_in_bytes
# are what we want. IDEA: read the limit outside the loop once, and check
# the usage every N iterations. GC only if usage > 9x% or so.
# to read file, use e.g. 
#   parse(Float64,read("/sys/fs/cgroup/user.slice/memory.min", String))
# It works regardless of whether there is a newline at the end or not

get_PBS_jobid() = haskey(ENV, "PBS_JOBID") ? ENV["PBS_JOBID"] : ""
function get_cgroup_mem_usage(jobid)
    fn = "/sys/fs/cgroup/memory/pbspro.service/jobid/$jobid/memory.usage_in_bytes"
    parse(Float64, read(fn, String))
end
function get_cgroup_mem_limit(jobid)
    fn = "/sys/fs/cgroup/memory/pbspro.service/jobid/$jobid/memory.limit_in_bytes"
    parse(Float64, read(fn, String))
end

# for the number of cpus available, need to parse
#   /sys/fs/cgroup/cpuset/pbspro.service/jobid/4274394.pbsha.ib.sockeye/cpuset.cpus
# see e.g.: https://github.com/cloudsigma/cgroupspy/blob/master/cgroupspy/interfaces.py#L177