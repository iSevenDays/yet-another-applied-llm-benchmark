## Copyright (C) 2024, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <http://www.gnu.org/licenses/>.

import asyncio
import pickle
import sys
import time
import tarfile
import io
import threading
import signal
import subprocess
import pty
import os
import select
import re
import termios
import struct
import fcntl
import random
import json
from config_loader import load_config
import re
import logging
import queue
import atexit


# DO NOT SET THIS FLAG TO TRUE UNLESS YOU ARE SURE YOU UNDERSTAND THE CONSEQUENCES
# IT IS VERY DANGEROUS. YOU WILL BE DIRECTLY EVALUATING WHATEVER COMES OUT OF
# A LANGUAGE MODEL DIRECTLY ON YOUR COMPUTER WITH NO SAFETY CHECKS.
I_HAVE_BLIND_FAITH_IN_LLMS_AND_AM_OKAY_WITH_THEM_BRICKING_MY_MACHINE_OR_MAKING_THEM_HALT_AND_CATCH_FIRE = False

BACKEND = load_config()['container']

# Container Pool for Performance Optimization
class ContainerPool:
    """
    Thread-safe container pool to eliminate setup/teardown overhead.
    Follows SPARC principles: Simple queue-based design that enhances existing architecture.
    """
    def __init__(self, max_size=4):
        self.max_size = max_size
        self.pool = queue.Queue(maxsize=max_size)
        self.active_containers = set()
        self.lock = threading.Lock()
        self._setup_cleanup()
        
    def _setup_cleanup(self):
        """Register cleanup function to ensure containers are destroyed on exit."""
        atexit.register(self._cleanup_all_containers)
        
    def _create_container(self):
        """Create a new container using existing backend logic."""
        if BACKEND == "docker":
            import docker
            docker_client = docker.from_env()
            container = docker_client.containers.run("llm-benchmark-image", detach=True, tty=True)
            return docker_client, container
        else:  # podman
            result = subprocess.run(["podman", "run", "-d", "-t", "llm-benchmark-image"], 
                                  capture_output=True, text=True, check=True)
            container_id = result.stdout.strip()
            return "PODMAN_CLIENT", container_id
            
    def _reset_container(self, docker_client, container):
        """Reset container state for reuse."""
        try:
            if BACKEND == "docker":
                # Clean up any files in the working directory
                container.exec_run(["rm", "-rf", "/usr/src/app/*"], detach=False)
                container.exec_run(["mkdir", "-p", "/usr/src/app"], detach=False)
            else:  # podman
                subprocess.run(["podman", "exec", container, "rm", "-rf", "/usr/src/app/*"], 
                             capture_output=True, check=False)
                subprocess.run(["podman", "exec", container, "mkdir", "-p", "/usr/src/app"], 
                             capture_output=True, check=True)
            return True
        except Exception as e:
            logging.warning(f"Failed to reset container: {e}")
            return False
    
    def get_container(self):
        """Get a container from the pool or create a new one."""
        try:
            # Try to get from pool (non-blocking)
            docker_client, container = self.pool.get_nowait()
            logging.debug("ContainerPool: Reusing container from pool")
            return docker_client, container
        except queue.Empty:
            # Pool is empty, create new container
            try:
                docker_client, container = self._create_container()
                with self.lock:
                    self.active_containers.add((docker_client, container))
                logging.debug("ContainerPool: Created new container")
                return docker_client, container
            except Exception as e:
                logging.error(f"ContainerPool: Failed to create container: {e}")
                raise
                
    def return_container(self, docker_client, container):
        """Return a container to the pool after resetting its state."""
        try:
            # Reset container state
            if self._reset_container(docker_client, container):
                # Try to return to pool (non-blocking)
                try:
                    self.pool.put_nowait((docker_client, container))
                    logging.debug("ContainerPool: Returned container to pool")
                except queue.Full:
                    # Pool is full, destroy this container
                    self._destroy_container(docker_client, container)
                    logging.debug("ContainerPool: Pool full, destroyed excess container")
            else:
                # Reset failed, destroy container
                self._destroy_container(docker_client, container)
                logging.debug("ContainerPool: Reset failed, destroyed container")
        except Exception as e:
            logging.error(f"ContainerPool: Error returning container: {e}")
            self._destroy_container(docker_client, container)
            
    def _destroy_container(self, docker_client, container):
        """Destroy a container using existing cleanup logic."""
        with self.lock:
            self.active_containers.discard((docker_client, container))
            
        if BACKEND == "docker":
            stop_and_remove_container(docker_client, container.id)
        else:  # podman
            stop_and_remove_podman_container(container)
            
    def _cleanup_all_containers(self):
        """Clean up all containers in pool and active set."""
        logging.info("ContainerPool: Cleaning up all containers")
        
        # Empty the pool and destroy containers
        while not self.pool.empty():
            try:
                docker_client, container = self.pool.get_nowait()
                self._destroy_container(docker_client, container)
            except queue.Empty:
                break
                
        # Destroy any remaining active containers
        with self.lock:
            for docker_client, container in list(self.active_containers):
                self._destroy_container(docker_client, container)

# Global container pool instance
_container_pool = None

def get_container_pool():
    """Get the global container pool instance."""
    global _container_pool
    if _container_pool is None:
        _container_pool = ContainerPool()
    return _container_pool

def make_tar(files):
    file_like_object = io.BytesIO()
    tar = tarfile.TarFile(fileobj=file_like_object, mode='w')
    
    for file_name, file_content in files.items():
        tarinfo = tarfile.TarInfo(name=file_name)
        tarinfo.size = len(file_content)
        tarinfo.mtime = time.time()
        tar.addfile(tarinfo, io.BytesIO(file_content))

    tar.close()

    file_like_object.seek(0)

    return file_like_object
    

if BACKEND == "docker":
    import docker
    def setup_docker(env):
        pool = get_container_pool()
        env.docker, env.container = pool.get_container()
        # Mark this environment as using pooled container
        env._using_pooled_container = True    
    
    def stop_and_remove_container(client, container_id):
        try:
            # Stopping the container with timeout to prevent hanging
            container = client.containers.get(container_id)
            container.stop(timeout=10)  # 10 second timeout
            container.remove()
        except Exception:
            # If graceful stop fails, force remove the container
            try:
                container = client.containers.get(container_id)
                container.remove(force=True)
            except Exception:
                # Container might already be stopped/removed, which is fine
                pass
    
    def async_kill_container(client, container):
        thread = threading.Thread(target=stop_and_remove_container, args=(client, container.id))
        thread.daemon = True
        thread.start()
        
    def return_container_to_pool(env):
        """Return a container to the pool instead of destroying it."""
        if hasattr(env, '_using_pooled_container') and env._using_pooled_container:
            pool = get_container_pool()
            pool.return_container(env.docker, env.container)
        else:
            # Fallback to old behavior for non-pooled containers
            async_kill_container(env.docker, env.container)
        
    
    def safe_run(client, container, files, run_cmd):
        tarfile = make_tar(files)
    
        path = "/usr/src/app"
        container.put_archive(path, tarfile)
    
        # Execute command in container (exec_run doesn't support timeout parameter)
        exit_code, output = container.exec_run(run_cmd)
        
        return output
elif BACKEND == "podman":
    def setup_docker(env):
        pool = get_container_pool()
        env.docker, env.container = pool.get_container()
        # Mark this environment as using pooled container
        env._using_pooled_container = True
    
    def stop_and_remove_podman_container(container_id):
        try:
            # Stopping the container with timeout to prevent hanging
            subprocess.run(["podman", "container", "stop", "--time", "10", container_id], 
                         timeout=15, check=True)
            # Removing the container
            subprocess.run(["podman", "container", "rm", container_id], 
                         timeout=10, check=True)
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
            # If graceful stop fails, force remove the container
            try:
                subprocess.run(["podman", "container", "rm", "--force", container_id], 
                             timeout=10, check=False)
            except Exception:
                # Container might already be stopped/removed, which is fine
                pass
    
    def async_kill_container(client, container_id):
        thread = threading.Thread(target=stop_and_remove_podman_container, args=(container_id,))
        thread.daemon = True
        thread.start()
        
    def return_container_to_pool(env):
        """Return a container to the pool instead of destroying it."""
        if hasattr(env, '_using_pooled_container') and env._using_pooled_container:
            pool = get_container_pool()
            pool.return_container(env.docker, env.container)
        else:
            # Fallback to old behavior for non-pooled containers
            async_kill_container(env.docker, env.container)
    
    def safe_run(client, container_id, files, run_cmd):
        tarfile = make_tar(files)

        # Create a temporary directory in the container to store files
        subprocess.run(["podman", "exec", container_id, "mkdir", "-p", "/usr/src/app"], check=True)
    
        # Copying files to the container
        r = random.randint(0, 1000000)
        with open('/tmp/archive%d.tar'%r, 'wb') as out_f:
            out_f.write(tarfile.getbuffer())
        time.sleep(.1)
        subprocess.run(["podman", "cp", "/tmp/archive%d.tar"%r, f"{container_id}:/usr/src/app"], check=True)
        time.sleep(.1)

        result = subprocess.run(["podman", "exec", container_id, "tar", "-xf", "archive%d.tar"%r], capture_output=True, check=True)

        time.sleep(.3)
        
        # Executing command in the container with timeout
        result = subprocess.run(["podman", "exec", container_id, *run_cmd], capture_output=True, timeout=30)
    
        return result.stdout + result.stderr
else:
    raise ValueError("Invalid backend")

import fcntl

def is_fd_closed(fd):
    try:
        fcntl.fcntl(fd, fcntl.F_GETFD)
        return False
    except OSError:
        return True


class DockerJob:
    def __init__(self, container_id, eos_string):
        self.eos_string = eos_string
        self.container_id = container_id

        if BACKEND == "docker":
            cmd = f"docker exec -i {container_id} /bin/bash"
            print("Running", cmd)
        else:
            cmd = f"podman exec -i {container_id} /bin/bash"
        
        self.process = subprocess.Popen(cmd,
                                        shell=True,
                                        stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE,
                                        text=True)
        
        self.master_fd = self.process.stdout.fileno()  # If you need a file descriptor for reading output
        
    @staticmethod
    def remove_ansi(text):
        ansi_escape =re.compile(r'(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]')
        return ansi_escape.sub('', text)
        
    def _detect_program_ready(self, cmd, total_output):
        """
        Smart detection for interactive program readiness.
        
        Some programs (like SQLite) don't emit prompts when using pipes,
        but are fully functional. This method detects readiness using
        program-specific strategies instead of relying only on prompt strings.
        """
        # For SQLite: Use probe-based detection
        if 'sqlite3' in cmd.lower():
            logging.debug(f"DockerJob: Detected SQLite, using probe-based readiness detection")
            return self._probe_sqlite_ready()
        
        # For other programs: Fall back to original EOS detection
        if self.eos_string in total_output:
            logging.debug(f"DockerJob: Found EOS string '{self.eos_string}' in output")
            return True
            
        return False
        
    def _probe_sqlite_ready(self):
        """
        Test if SQLite is ready by sending a probe command and checking response.
        
        SQLite doesn't show 'sqlite>' prompt with pipes, but responds to commands.
        We send '.help' and look for characteristic SQLite help output.
        """
        try:
            logging.debug("DockerJob: Probing SQLite readiness with .help command")
            
            # Send probe command
            self.process.stdin.write('.help\n')
            self.process.stdin.flush()
            
            # Wait for response with timeout using non-blocking read
            ready, _, _ = select.select([self.master_fd], [], [], 3)
            if ready:
                # Non-blocking read to avoid hanging
                import os
                response = os.read(self.master_fd, 1024).decode('utf-8', errors='ignore')
                logging.debug(f"DockerJob: SQLite probe response: {repr(response[:100])}")
                
                # Look for characteristic SQLite help output
                if '.archive' in response or '.backup' in response or '.help' in response:
                    logging.info("DockerJob: SQLite is ready and responsive")
                    return True
                else:
                    logging.debug(f"DockerJob: SQLite responded but no help output: {repr(response[:50])}")
            else:
                logging.debug("DockerJob: SQLite probe timed out - trying without probe")
                
        except Exception as e:
            logging.debug(f"DockerJob: SQLite probe failed: {e}")
            
        return False
        
    def _cleanup_process(self):
        """Clean up any hanging processes in the container."""
        try:
            if BACKEND == "docker":
                # Kill any SQLite processes that might be hanging
                subprocess.run(["docker", "exec", self.container_id, "pkill", "-f", "sqlite3"], 
                             capture_output=True, check=False)
            else:
                subprocess.run(["podman", "exec", self.container_id, "pkill", "-f", "sqlite3"], 
                             capture_output=True, check=False)
        except Exception as e:
            logging.debug(f"DockerJob: Process cleanup attempt: {e}")

    def __call__(self, cmd):
        # Check if process is still alive
        if self.process.poll() is not None:
            print(f"Process has terminated with return code: {self.process.returncode}")
            return f"Process terminated (exit code: {self.process.returncode})"
            
        # Send the command through the PTY
        print("GO", self.process.stdin)
        try:
            # Ensure cmd is a string and properly formatted
            if not isinstance(cmd, str):
                cmd = str(cmd)
            
            # Clean up the command - remove any problematic characters
            cmd = cmd.strip()
            if not cmd:
                cmd = ""  # Empty command
                
            command_to_send = cmd + "\n"
            
            # Check if stdin is still open
            if self.process.stdin.closed:
                print("Process stdin is closed")
                return "Process stdin is closed"
                
            self.process.stdin.write(command_to_send)
            self.process.stdin.flush()
            print(f"Sent command: {repr(command_to_send)}")
        except BrokenPipeError:
            print("Broken pipe - process may have terminated")
            return "Broken pipe - process may have terminated"
        except Exception as e:
            print(f"Process communication failed: {e}")
            return f"Process communication failed: {e}"

        # Read the output with smart detection for program readiness
        output = []
        timeout_count = 0
        
        # Adaptive timeout settings based on command type
        if 'sqlite3' in cmd.lower():
            max_timeouts = 2  # SQLite should respond quickly to probe
            timeout_duration = 4  # Shorter timeouts for probe-based detection
            logging.debug("DockerJob: Using SQLite-optimized timeouts")
        else:
            max_timeouts = 3  # Standard timeout for other programs
            timeout_duration = 10  # Standard timeout duration
            
        total_output = ""  # Track accumulated output for detection
        program_ready = False
        
        # First, try smart detection immediately for programs that might be ready
        if 'sqlite3' in cmd.lower():
            # Give SQLite a moment to start, then test readiness
            import time
            time.sleep(0.5)
            if self._detect_program_ready(cmd, total_output):
                program_ready = True
                logging.info("DockerJob: Program detected as ready via smart detection")
        
        while not program_ready:
            ready, _, _ = select.select([self.master_fd], [], [], timeout_duration)
            if ready:
                timeout_count = 0  # Reset timeout counter on successful read
                try:
                    # Use non-blocking read to prevent hangs
                    import os
                    line = os.read(self.master_fd, 128).decode('utf-8', errors='ignore')
                    if line:
                        output.append(line)
                        total_output += line
                        logging.debug(f"DockerJob received chunk ({len(line)} chars): {repr(line[:50])}")
                        
                        # Check if program is ready using smart detection
                        if self._detect_program_ready(cmd, total_output):
                            logging.debug("DockerJob: Program ready detected")
                            program_ready = True
                            break
                    else:
                        # Empty read indicates end of stream
                        logging.debug("DockerJob: Empty read - end of stream")
                        break
                except (OSError, ValueError) as e:
                    logging.error(f"DockerJob error reading from process stdout: {e}")
                    break
            else:
                # Timeout occurred
                timeout_count += 1
                logging.debug(f"DockerJob accumulated output so far ({len(total_output)} chars): {repr(total_output[:100])}")
                
                # For interactive programs, try smart detection on timeout
                if self._detect_program_ready(cmd, total_output):
                    logging.info("DockerJob: Program ready detected after timeout")
                    program_ready = True
                    break
                
                if timeout_count >= max_timeouts:
                    logging.warning(f"DockerJob timeout - no response after {max_timeouts} attempts ({timeout_duration}s each)")
                    logging.warning(f"DockerJob expected EOS: {repr(self.eos_string)}, got: {repr(total_output)}")
                    
                    # Clean up any hanging processes
                    self._cleanup_process()
                    break
                else:
                    logging.debug(f"DockerJob timeout #{timeout_count} - retrying...")
                    continue


        output = ''.join(output)
        output = self.remove_ansi(output)
        print("Output:", repr(output))
        return output


def invoke_docker(env, files, run_cmd, out_bytes=False):
    if env.docker is None:
        setup_docker(env)

    def raise_timeout(signum, frame):
        raise TimeoutError
    signal.signal(signal.SIGALRM, raise_timeout)
    signal.alarm(50)
    
    try:
        # Function call that might take too long
        out = safe_run(env.docker, env.container, files, run_cmd)
    except TimeoutError:
        out = b"Timeout: function took too long to complete"

    signal.alarm(0) 

    if out_bytes:
        return out
    else:
        return out.decode("utf-8")


if I_HAVE_BLIND_FAITH_IN_LLMS_AND_AM_OKAY_WITH_THEM_BRICKING_MY_MACHINE_OR_MAKING_THEM_HALT_AND_CATCH_FIRE:

    class DockerJob:
        def __init__(self, container_id, eos_string):
            raise NotImplementedError("This test is not implemented in unsafe mode yet")
        
    def setup_docker(env):
        import random
        env.fake_docker_id = random.randint(0, 10000000000)
        os.mkdir("/tmp/fakedocker_%d"%env.fake_docker_id)
        

    def invoke_docker(env, files, run_cmd, out_bytes=False):
        if env.docker is None:
            setup_docker(env)
    
        def raise_timeout(signum, frame):
            raise TimeoutError
        signal.signal(signal.SIGALRM, raise_timeout)
        signal.alarm(50)
        
        try:
            # Function call that might take too long
            for file_name, file_content in files.items():
                with open("/tmp/fakedocker_%d/%s"%(env.fake_docker_id, file_name), "wb") as f:
                    f.write(file_content)
            proc = subprocess.run(run_cmd, cwd="/tmp/fakedocker_%d"%env.fake_docker_id, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except TimeoutError:
            if out_bytes:
                return b"Timeout: function took too long to complete"
            else:
                return "Timeout: function took too long to complete"

        signal.alarm(0) 
    

        if out_bytes:
            return proc.stdout + proc.stderr
        else:
            stdout = proc.stdout.decode("utf-8")
            stderr = proc.stderr.decode("utf-8")
            
            # Replace /fakedocker_[0-9]*/ with /fakedocker/
            stdout = re.sub(r'/fakedocker_[0-9]*/', '/fakedocker/', stdout)
            stderr = re.sub(r'/fakedocker_[0-9]*/', '/fakedocker/', stderr)
        
            return stdout + stderr
