"""Plain-language names, units, and descriptions for each raw NAB series filename.
Keeps the dashboard data-oriented: a professor sees 'CPU usage - Server 1 (%)', not a hash."""

# filename -> (friendly label, unit, one-line description)
META = {
    "ec2_cpu_utilization_24ae8d.csv": ("CPU usage - Server 1", "%", "How hard the server's processor is working."),
    "ec2_cpu_utilization_53ea38.csv": ("CPU usage - Server 2", "%", "How hard the server's processor is working."),
    "ec2_cpu_utilization_5f5533.csv": ("CPU usage - Server 3", "%", "How hard the server's processor is working."),
    "ec2_cpu_utilization_77c1ca.csv": ("CPU usage - Server 4", "%", "How hard the server's processor is working."),
    "ec2_cpu_utilization_825cc2.csv": ("CPU usage - Server 5", "%", "How hard the server's processor is working."),
    "ec2_cpu_utilization_ac20cd.csv": ("CPU usage - Server 6", "%", "How hard the server's processor is working."),
    "ec2_cpu_utilization_c6585a.csv": ("CPU usage - Server 7", "%", "How hard the server's processor is working."),
    "ec2_cpu_utilization_fe7f93.csv": ("CPU usage - Server 8", "%", "How hard the server's processor is working."),
    "ec2_disk_write_bytes_1ef3de.csv": ("Disk writes - Server 1", "bytes", "How much data the server is saving to storage."),
    "ec2_disk_write_bytes_c0d644.csv": ("Disk writes - Server 2", "bytes", "How much data the server is saving to storage."),
    "ec2_network_in_257a54.csv": ("Network in - Server 1", "bytes", "How much data is arriving over the network."),
    "ec2_network_in_5abac7.csv": ("Network in - Server 2", "bytes", "How much data is arriving over the network."),
    "elb_request_count_8c0756.csv": ("Web requests - Load balancer", "requests", "How many web requests are hitting the service."),
    "grok_asg_anomaly.csv": ("Auto-scaling group load", "units", "Overall load across a group of servers."),
    "iio_us-east-1_i-a2eb1cd9_NetworkIn.csv": ("Network in - Server 3", "bytes", "How much data is arriving over the network."),
    "rds_cpu_utilization_cc0c53.csv": ("Database CPU usage - DB 1", "%", "How hard the database server is working."),
    "rds_cpu_utilization_e47b3b.csv": ("Database CPU usage - DB 2", "%", "How hard the database server is working."),
}


def label(fname):
    return META.get(fname, (fname, "", ""))[0]


def unit(fname):
    return META.get(fname, ("", "", ""))[1]


def desc(fname):
    return META.get(fname, ("", "", ""))[2]


if __name__ == "__main__":
    from data import all_series
    missing = [s for s in all_series() if s not in META]
    assert not missing, f"unmapped series: {missing}"
    print(f"all {len(all_series())} series have plain-language metadata")
