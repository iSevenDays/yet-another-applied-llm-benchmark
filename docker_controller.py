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
DOCKER_EXEC_TIMEOUT = 50

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
        build_path = os.path.dirname(os.path.abspath(__file__))
        if BACKEND == "docker":
            import docker
            docker_client = docker.from_env()
            try:
                docker_client.images.get("llm-benchmark-image")
            except docker.errors.ImageNotFound:
                logging.info("ContainerPool: llm-benchmark-image not found locally, building from Dockerfile...")
                docker_client.images.build(path=build_path, tag="llm-benchmark-image")
            container = docker_client.containers.run("llm-benchmark-image", detach=True, tty=True)
            return docker_client, container
        else:  # podman
            result = subprocess.run(["podman", "images", "-q", "llm-benchmark-image"],
                                  capture_output=True, text=True)
            if not result.stdout.strip():
                logging.info("ContainerPool: llm-benchmark-image not found locally, building from Dockerfile...")
                subprocess.run(["podman", "build", "-t", "llm-benchmark-image", build_path],
                              check=True)
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
        self._sqlite_mode = False  # Set during first call if command is sqlite3

        if BACKEND == "docker":
            cmd = f"docker exec -i {container_id} /bin/bash"
            logging.debug("DockerJob: Running %s", cmd)
        else:
            cmd = f"podman exec -i {container_id} /bin/bash"

        self.process = subprocess.Popen(cmd,
                                        shell=True,
                                        stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE,
                                        text=True)

        self.master_fd = self.process.stdout.fileno()
        
    @staticmethod
    def remove_ansi(text):
        ansi_escape =re.compile(r'(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]')
        return ansi_escape.sub('', text)
        
    def _is_sqlite_session(self):
        """Check if this DockerJob is running SQLite."""
        return self._sqlite_mode

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

    def _send_command(self, cmd):
        """Send a command string to the process stdin. Returns error string or None."""
        if not isinstance(cmd, str):
            cmd = str(cmd)
        cmd = cmd.strip()
        command_to_send = cmd + "\n"

        if self.process.stdin.closed:
            logging.warning("DockerJob: Process stdin is closed")
            return "Process stdin is closed"

        try:
            self.process.stdin.write(command_to_send)
            self.process.stdin.flush()
            logging.debug("DockerJob: Sent command: %s", repr(command_to_send))
        except BrokenPipeError:
            logging.warning("DockerJob: Broken pipe - process may have terminated")
            return "Broken pipe - process may have terminated"
        except Exception as e:
            logging.error("DockerJob: Process communication failed: %s", e)
            return f"Process communication failed: {e}"
        return None

    def _read_until_idle(self, timeout_per_read, max_idle_rounds):
        """Read output until no data arrives for max_idle_rounds consecutive timeouts."""
        output = []
        idle_count = 0

        while idle_count < max_idle_rounds:
            ready, _, _ = select.select([self.master_fd], [], [], timeout_per_read)
            if ready:
                idle_count = 0
                try:
                    chunk = os.read(self.master_fd, 4096).decode('utf-8', errors='ignore')
                    if chunk:
                        output.append(chunk)
                        logging.debug(f"DockerJob read {len(chunk)} chars")
                    else:
                        break  # EOF
                except (OSError, ValueError) as e:
                    logging.error(f"DockerJob read error: {e}")
                    break
            else:
                idle_count += 1

        return ''.join(output)

    def __call__(self, cmd):
        if self.process.poll() is not None:
            logging.warning("DockerJob: Process terminated with return code: %s", self.process.returncode)
            return f"Process terminated (exit code: {self.process.returncode})"

        # Detect if this is the initial SQLite startup command
        is_startup = 'sqlite3' in cmd.lower()
        if is_startup:
            self._sqlite_mode = True

        err = self._send_command(cmd)
        if err:
            return err

        if is_startup:
            # SQLite startup: wait briefly for it to initialize, drain any banner output.
            # SQLite in pipe mode doesn't show a prompt, so we just wait until idle.
            import time
            time.sleep(0.5)
            startup_output = self._read_until_idle(timeout_per_read=1.0, max_idle_rounds=2)
            logging.debug("DockerJob: SQLite startup output: %s", repr(startup_output[:200]))
            return self.remove_ansi(startup_output)

        if self._sqlite_mode:
            # Subsequent SQL commands: SQLite executes instantly, read with short timeouts
            output = self._read_until_idle(timeout_per_read=2.0, max_idle_rounds=2)
            return self.remove_ansi(output)

        # Non-SQLite programs: use EOS string detection with standard timeouts
        output = []
        total_output = ""
        timeout_count = 0
        max_timeouts = 3
        timeout_duration = 10

        while True:
            ready, _, _ = select.select([self.master_fd], [], [], timeout_duration)
            if ready:
                timeout_count = 0
                try:
                    chunk = os.read(self.master_fd, 4096).decode('utf-8', errors='ignore')
                    if chunk:
                        output.append(chunk)
                        total_output += chunk
                        if self.eos_string in total_output:
                            logging.debug("DockerJob: Found EOS string")
                            break
                    else:
                        break
                except (OSError, ValueError) as e:
                    logging.error(f"DockerJob read error: {e}")
                    break
            else:
                timeout_count += 1
                if timeout_count >= max_timeouts:
                    logging.warning(f"DockerJob timeout after {max_timeouts} x {timeout_duration}s")
                    self._cleanup_process()
                    break

        result = ''.join(output)
        return self.remove_ansi(result)


def _save_alarm_state():
    """Save current SIGALRM state for nesting support in parallel mode."""
    remaining = signal.alarm(0)
    handler = signal.getsignal(signal.SIGALRM)
    return (remaining, handler, time.time())


def _restore_alarm_state(state):
    """Restore previously saved SIGALRM state, adjusting for elapsed time."""
    prev_remaining, prev_handler, start_time = state
    if prev_handler not in (signal.SIG_DFL, signal.SIG_IGN, None):
        signal.signal(signal.SIGALRM, prev_handler)
        if prev_remaining > 0:
            elapsed = int(time.time() - start_time)
            signal.alarm(max(1, prev_remaining - elapsed))


def invoke_docker(env, files, run_cmd, out_bytes=False):
    if env.docker is None:
        setup_docker(env)

    # Save previous alarm state to support nesting with worker-level timeouts
    alarm_state = _save_alarm_state()
    prev_remaining = alarm_state[0]
    effective_timeout = min(DOCKER_EXEC_TIMEOUT, prev_remaining) if prev_remaining > 0 else DOCKER_EXEC_TIMEOUT

    def raise_timeout(signum, frame):
        raise TimeoutError
    signal.signal(signal.SIGALRM, raise_timeout)
    signal.alarm(effective_timeout)

    try:
        out = safe_run(env.docker, env.container, files, run_cmd)
    except TimeoutError:
        out = b"Timeout: function took too long to complete"
    finally:
        signal.alarm(0)
        _restore_alarm_state(alarm_state)

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

        # Save previous alarm state to support nesting with worker-level timeouts
        alarm_state = _save_alarm_state()
        prev_remaining = alarm_state[0]
        effective_timeout = min(DOCKER_EXEC_TIMEOUT, prev_remaining) if prev_remaining > 0 else DOCKER_EXEC_TIMEOUT

        def raise_timeout(signum, frame):
            raise TimeoutError
        signal.signal(signal.SIGALRM, raise_timeout)
        signal.alarm(effective_timeout)

        timed_out = False
        try:
            for file_name, file_content in files.items():
                with open("/tmp/fakedocker_%d/%s"%(env.fake_docker_id, file_name), "wb") as f:
                    f.write(file_content)
            proc = subprocess.run(run_cmd, cwd="/tmp/fakedocker_%d"%env.fake_docker_id, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except TimeoutError:
            timed_out = True
        finally:
            signal.alarm(0)
            _restore_alarm_state(alarm_state)

        if timed_out:
            if out_bytes:
                return b"Timeout: function took too long to complete"
            else:
                return "Timeout: function took too long to complete"

        if out_bytes:
            return proc.stdout + proc.stderr
        else:
            stdout = proc.stdout.decode("utf-8")
            stderr = proc.stderr.decode("utf-8")

            # Replace /fakedocker_[0-9]*/ with /fakedocker/
            stdout = re.sub(r'/fakedocker_[0-9]*/', '/fakedocker/', stdout)
            stderr = re.sub(r'/fakedocker_[0-9]*/', '/fakedocker/', stderr)

            return stdout + stderr
