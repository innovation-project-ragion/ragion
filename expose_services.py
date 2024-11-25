import subprocess
import time

def expose_tunnel(port, service_name):
    """
    Create a tunnel for a specific service and port using py-localtunnel CLI.
    """
    print(f"Exposing {service_name} on port {port}...")
    try:
        # Run the pylt command to open a tunnel
        result = subprocess.run(
            ['pylt', 'port', str(port)],
            capture_output=True,
            text=True,
            check=True
        )
        # Extract the public URL from the command output
        output_lines = result.stdout.splitlines()
        for line in output_lines:
            if 'url' in line:
                public_url = line.split(' ')[-1]
                print(f"{service_name} is now accessible at: {public_url}")
                return public_url
        print(f"Failed to retrieve URL for {service_name}.")
        return None
    except subprocess.CalledProcessError as e:
        print(f"Error exposing {service_name}: {e.stderr}")
        return None

def main():
    print("Skipping Docker Compose startup since it's already running...")

    # Services to expose and their corresponding ports
    services = {
        "Milvus gRPC": 19530,
        "Milvus Health Check": 9091,
        "Attu": 3000,
        "Neo4j Browser": 7474,
        "Neo4j Bolt Protocol": 7687,
    }

    exposed_urls = {}

    for service_name, port in services.items():
        url = expose_tunnel(port, service_name)
        if url:
            exposed_urls[service_name] = url

    # Display all exposed URLs
    print("\nAll services exposed:")
    for service, url in exposed_urls.items():
        print(f"{service}: {url}")

    # Keep the script running to maintain tunnels
    print("\nTunnels are active. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down tunnels...")

if __name__ == "__main__":
    main()
