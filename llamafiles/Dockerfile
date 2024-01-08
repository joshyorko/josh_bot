FROM ubuntu:focal as base

# Install Python and pip
RUN apt-get update && apt-get install -y python3 python3-pip

# Copy the script into the container
COPY ./model_dir/mistral-7b-instruct-v0.1-Q4_K_M-server.llamafile /model_dir/mistral-7b-instruct-v0.1-Q4_K_M-server.llamafile

# Change the permissions of the script so it can be executed
RUN chmod +x /model_dir/mistral-7b-instruct-v0.1-Q4_K_M-server.llamafile

CMD ["./model_dir/mistral-7b-instruct-v0.1-Q4_K_M-server.llamafile", "--ngl", "5"]