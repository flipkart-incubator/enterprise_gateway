# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.
"""Kernel managers that operate against a remote process."""

import os
import signal
import json
import time
import logging
import subprocess
import errno
import socket
from jupyter_client import launch_kernel, localinterfaces
from .processproxy import RemoteProcessProxy
from yarn_api_client.resource_manager import ResourceManager
try:
    from urlparse import urlparse
except ImportError:
    from urllib.parse import urlparse

# Default logging level of yarn-api produces too much noise - raise to warning only.
logging.getLogger('yarn_api_client.resource_manager').setLevel(os.getenv('EG_YARN_LOG_LEVEL', logging.WARNING))

local_ip = localinterfaces.public_ips()[0]
poll_interval = float(os.getenv('EG_POLL_INTERVAL', '0.5'))
max_poll_attempts = int(os.getenv('EG_MAX_POLL_ATTEMPTS', '10'))
yarn_shutdown_wait_time = float(os.getenv('EG_YARN_SHUTDOWN_WAIT_TIME', '15.0'))


class YarnClusterProcessProxy(RemoteProcessProxy):
    initial_states = {'NEW', 'SUBMITTED', 'ACCEPTED', 'RUNNING'}
    final_states = {'FINISHED', 'KILLED'}  # Don't include FAILED state

    def __init__(self, kernel_manager, proxy_config):
        super(YarnClusterProcessProxy, self).__init__(kernel_manager, proxy_config)
        self.application_id = None
        self.yarn_endpoint \
            = proxy_config.get('yarn_endpoint',
                               kernel_manager.parent.parent.yarn_endpoint)
        self.yarn_endpoint_security_enabled \
            = proxy_config.get('yarn_endpoint_security_enabled',
                               kernel_manager.parent.parent.yarn_endpoint_security_enabled)
        # #TODO remove this
        # self.yarn_endpoint = "http://10.32.77.243:8088"
        yarn_url = urlparse(self.yarn_endpoint)
        yarn_master = yarn_url.hostname
        yarn_port = yarn_url.port
        if self.yarn_endpoint_security_enabled is True:
            self.resource_mgr = ResourceManager(address=yarn_master,
                                                port=yarn_port,
                                                kerberos_enabled=self.yarn_endpoint_security_enabled)
        else:
            self.resource_mgr = ResourceManager(address=yarn_master,
                                                port=yarn_port)
        # YARN applications tend to take longer than the default 5 second wait time.  Rather than
        # require a command-line option for those using YARN, we'll adjust based on a local env that
        # defaults to 15 seconds.  Note: we'll only adjust if the current wait time is shorter than
        # the desired value.
        if kernel_manager.shutdown_wait_time < yarn_shutdown_wait_time:
            kernel_manager.shutdown_wait_time = yarn_shutdown_wait_time
            self.log.debug("{class_name} shutdown wait time adjusted to {wait_time} seconds.".
                           format(class_name=type(self).__name__, wait_time=kernel_manager.shutdown_wait_time))

        self.yarn_resource_check_wait_time = 0.20 * self.kernel_launch_timeout

    def launch_process(self, kernel_cmd, **kwargs):
        env_dict = kwargs.get('env')

        # checks to see if the queue resource is available
        # if not kernel startup is not tried
        self.confirm_yarn_queue_avaibility(env_dict)
        # Launches the specified process within a YARN cluster environment.
        super(YarnClusterProcessProxy, self).launch_process(kernel_cmd, **kwargs)

        # launch the local run.sh - which is configured for yarn-cluster...
        self.local_proc = launch_kernel(kernel_cmd, **kwargs)
        self.pid = self.local_proc.pid
        self.ip = local_ip

        self.log.debug("Yarn cluster kernel launched using YARN endpoint: {}, pid: {}, Kernel ID: {}, cmd: '{}'"
                       .format(self.yarn_endpoint, self.local_proc.pid, self.kernel_id, kernel_cmd))
        self.confirm_remote_startup(kernel_cmd, **kwargs)

        return self

    """submitting jobs to yarn queue and then checking till the jobs are in running state 
       will lead to orphan jobs being created in some scenarios.
       We take kernel_launch_timeout time and divide this into two parts.
       if the queue is unavailable we take max 20% of the time to poll the queue periodically
       and if the queue becomes available the rest of timeout is met in 80% of the remmaining 
       time.  

    """

    def confirm_yarn_queue_avaibility(self, env_dict):
        """
        confirms if the yarn queue has capacity to handle the resource requests that
        will be sent to it.
        if not, proper error messages are sent back for user experience
        :param env_dict:
        :return:
        """

        executer_memory = int(env_dict['EXECUTOR_MEMORY'])
        driver_memory = int(env_dict['DRIVER_MEMORY'])

        candidate_queue_name = env_dict['QUEUE']

        node_label = env_dict['NODE_LABEL']

        self.container_memory = self.resource_mgr.cluster_node_container_memory()
        self.candidate_queue = self.resource_mgr.cluster_scheduler_queue(candidate_queue_name)
        self.candidate_partition = self.resource_mgr.cluster_queue_partition(self.candidate_queue, node_label)


        if max(executer_memory, driver_memory) > self.container_memory:
            self.log_and_raise(http_status_code=500,
                               reason="Container Memory not sufficient for a executor/driver allocation")

        # else the resources may or may not be available now. it may be possible that if we wait then the resources
        # become available. start  a timeout process

        self.start_time = RemoteProcessProxy.get_current_time()
        yarn_available = False

        while not yarn_available:
            self.handle_yarn_queue_timeout()
            yarn_available = self.resource_mgr.cluster_scheduler_queue_availability(self.candidate_partition)

    def handle_yarn_queue_timeout(self):

        # each time we sleep we substract equal amount of time for the kernel launch timeout
        self.log.info("Kernel Launch Timeout - {}".format(self.kernel_launch_timeout))
        time.sleep(poll_interval)
        self.kernel_launch_timeout -= poll_interval

        time_interval = RemoteProcessProxy.get_time_diff(self.start_time, RemoteProcessProxy.get_current_time())

        if time_interval > self.yarn_resource_check_wait_time:
            error_http_code = 500
            reason = "Yarn Compute Resource is unavailable"
            self.log_and_raise(http_status_code=error_http_code, reason=reason)

    def poll(self):
        """Submitting a new kernel/app to YARN will take a while to be ACCEPTED.
        Thus application ID will probably not be available immediately for poll.
        So will regard the application as RUNNING when application ID still in ACCEPTED or SUBMITTED state.

        :return: None if the application's ID is available and state is ACCEPTED/SUBMITTED/RUNNING. Otherwise False.
        """
        result = False

        if self.get_application_id():
            state = self.query_app_state_by_id(self.application_id)
            if state in YarnClusterProcessProxy.initial_states:
                result = None

        # The following produces too much output (every 3 seconds by default), so commented-out at this time.
        # self.log.debug("YarnProcessProxy.poll, application ID: {}, kernel ID: {}, state: {}".
        #               format(self.application_id, self.kernel_id, state))
        return result

    def send_signal(self, signum):
        """Currently only support 0 as poll and other as kill.

        :param signum
        :return:
        """
        self.log.debug("YarnClusterProcessProxy.send_signal {}".format(signum))
        if signum == 0:
            return self.poll()
        elif signum == signal.SIGKILL:
            return self.kill()
        else:
            # Yarn api doesn't support the equivalent to interrupts, so take our chances
            # via a remote signal.  Note that this condition cannot check against the
            # signum value because altternate interrupt signals might be in play.
            return super(YarnClusterProcessProxy, self).send_signal(signum)

    def kill(self):
        """Kill a kernel.
        :return: None if the application existed and is not in RUNNING state, False otherwise.
        """
        state = None
        result = False
        if self.get_application_id():
            resp = self.kill_app_by_id(self.application_id)
            self.log.debug(
                "YarnClusterProcessProxy.kill: kill_app_by_id({}) response: {}, confirming app state is not RUNNING"
                    .format(self.application_id, resp))

            i = 1
            state = self.query_app_state_by_id(self.application_id)
            while state not in YarnClusterProcessProxy.final_states and i <= max_poll_attempts:
                time.sleep(poll_interval)
                state = self.query_app_state_by_id(self.application_id)
                i = i + 1

            if state in YarnClusterProcessProxy.final_states:
                result = None

        super(YarnClusterProcessProxy, self).kill()

        self.log.debug("YarnClusterProcessProxy.kill, application ID: {}, kernel ID: {}, state: {}"
                       .format(self.application_id, self.kernel_id, state))
        return result

    def cleanup(self):
        # we might have a defunct process (if using waitAppCompletion = false) - so poll, kill, wait when we have
        # a local_proc.
        if self.local_proc:
            self.log.debug("YarnClusterProcessProxy.cleanup: Clearing possible defunct process, pid={}...".
                           format(self.local_proc.pid))
            if super(YarnClusterProcessProxy, self).poll():
                super(YarnClusterProcessProxy, self).kill()
            super(YarnClusterProcessProxy, self).wait()
            self.local_proc = None

        # reset application id to force new query - handles kernel restarts/interrupts
        self.application_id = None

        # for cleanup, we should call the superclass last
        super(YarnClusterProcessProxy, self).cleanup()

    def confirm_remote_startup(self, kernel_cmd, **kw):
        """ Confirms the yarn application is in a started state before returning.  Should post-RUNNING states be
            unexpectedly encountered (FINISHED, KILLED) then we must throw, otherwise the rest of the JKG will
            believe its talking to a valid kernel.
        """
        self.start_time = RemoteProcessProxy.get_current_time()
        i = 0
        ready_to_connect = False  # we're ready to connect when we have a connection file to use
        while not ready_to_connect:
            i += 1
            self.handle_timeout()

            if self.get_application_id(True):
                # Once we have an application ID, start monitoring state, obtain assigned host and get connection info
                app_state = self.get_application_state()

                if app_state in YarnClusterProcessProxy.final_states:
                    error_message = "KernelID: '{}', ApplicationID: '{}' unexpectedly found in " \
                                                     "state '{}' during kernel startup!".\
                                    format(self.kernel_id, self.application_id, app_state)
                    self.log_and_raise(http_status_code=500, reason=error_message)

                self.log.debug("{}: State: '{}', Host: '{}', KernelID: '{}', ApplicationID: '{}'".
                               format(i, app_state, self.assigned_host, self.kernel_id, self.application_id))

                if self.assigned_host != '':
                    ready_to_connect = self.receive_connection_info()
            else:
                self.detect_launch_failure()

    def get_application_state(self):
        # Gets the current application state using the application_id already obtained.  Once the assigned host
        # has been identified, it is nolonger accessed.
        app_state = None
        app = self.query_app_by_id(self.application_id)

        if app:
            if app.get('state'):
                app_state = app.get('state')
            if self.assigned_host == '' and app.get('amHostHttpAddress'):
                self.assigned_host = app.get('amHostHttpAddress').split(':')[0]
                # Set the kernel manager ip to the actual host where the application landed.
                self.assigned_ip = socket.gethostbyname(self.assigned_host)
        return app_state

    def handle_timeout(self):
        time.sleep(poll_interval)
        time_interval = RemoteProcessProxy.get_time_diff(self.start_time, RemoteProcessProxy.get_current_time())

        if time_interval > self.kernel_launch_timeout:
            reason = "Application ID is None. Failed to submit a new application to YARN within {} seconds.  " \
                     "Check Enterprise Gateway log for more information.". \
                format(self.kernel_launch_timeout)
            error_http_code = 500
            if self.get_application_id(True):
                if self.query_app_state_by_id(self.application_id) != "RUNNING":
                    reason = "YARN resources unavailable after {} seconds for app {}, launch timeout: {}!  "\
                        "Check YARN configuration.".format(time_interval, self.application_id, self.kernel_launch_timeout)
                    error_http_code = 503
                else:
                    reason = "App {} is RUNNING, but waited too long ({} secs) to get connection file.  " \
                        "Check YARN logs for more information.".format(self.application_id, self.kernel_launch_timeout)
            self.kill()
            timeout_message = "KernelID: '{}' launch timeout due to: {}".format(self.kernel_id, reason)
            self.log_and_raise(http_status_code=error_http_code, reason=timeout_message)

    def get_application_id(self, ignore_final_states=False):
        # Return the kernel's YARN application ID if available, otherwise None.  If we're obtaining application_id
        # from scratch, do not consider kernels in final states.
        if not self.application_id:
            app = self.query_app_by_name(self.kernel_id)
            state_condition = True
            if type(app) is dict and ignore_final_states:
                state_condition = app.get('state') not in YarnClusterProcessProxy.final_states

            if type(app) is dict and len(app.get('id', '')) > 0 and state_condition:
                self.application_id = app['id']
                time_interval = RemoteProcessProxy.get_time_diff(self.start_time, RemoteProcessProxy.get_current_time())
                self.log.info("ApplicationID: '{}' assigned for KernelID: '{}', state: {}, {} seconds after starting."
                              .format(app['id'], self.kernel_id, app.get('state'), time_interval))
            else:
                self.log.debug("ApplicationID not yet assigned for KernelID: '{}' - retrying...".format(self.kernel_id))
        return self.application_id

    def get_process_info(self):
        process_info = super(YarnClusterProcessProxy, self).get_process_info()
        process_info.update({'application_id': self.application_id})
        return process_info

    def load_process_info(self, process_info):
        super(YarnClusterProcessProxy, self).load_process_info(process_info)
        self.application_id = process_info['application_id']

    def query_app_by_name(self, kernel_id):
        """Retrieve application by using kernel_id as the unique app name.
        With the started_time_begin as a parameter to filter applications started earlier than the target one from YARN.
        When submit a new app, it may take a while for YARN to accept and run and generate the application ID.
        Note: if a kernel restarts with the same kernel id as app name, multiple applications will be returned.
        For now, the app/kernel with the top most application ID will be returned as the target app, assuming the app
        ID will be incremented automatically on the YARN side.

        :param kernel_id: as the unique app name for query
        :return: The JSON object of an application.
        """
        top_most_app_id = ''
        target_app = None
        data = None
        try:
            data = self.resource_mgr.cluster_applications(started_time_begin=str(self.start_time)).data
        except socket.error as sock_err:
            if sock_err.errno == errno.ECONNREFUSED:
                self.log.warning("YARN end-point: '{}' refused the connection.  Is the resource manager running?".
                               format(self.yarn_endpoint))
            else:
                self.log.warning("Query for kernel ID '{}' failed with exception: {} - '{}'.  Continuing...".
                                 format(kernel_id, type(sock_err), sock_err))
        except Exception as e:
            self.log.warning("Query for kernel ID '{}' failed with exception: {} - '{}'.  Continuing...".
                             format(kernel_id, type(e), e))

        if type(data) is dict and type(data.get("apps")) is dict and 'app' in data.get("apps"):
            for app in data['apps']['app']:
                if app.get('name', '').find(kernel_id) >= 0 and app.get('id') > top_most_app_id:
                    target_app = app
                    top_most_app_id = app.get('id')
        return target_app

    def query_app_by_id(self, app_id):
        """Retrieve an application by application ID.

        :param app_id
        :return: The JSON object of an application.
        """
        data = None
        try:
            data = self.resource_mgr.cluster_application(application_id=app_id).data
        except Exception as e:
            self.log.warning("Query for application ID '{}' failed with exception: '{}'.  Continuing...".
                           format(app_id, e))
        if type(data) is dict and 'app' in data:
            return data['app']
        return None

    def query_app_state_by_id(self, app_id):
        """Return the state of an application.
        :param app_id:
        :return:
        """
        response = None
        try:
            response = self.resource_mgr.cluster_application_state(application_id=app_id)
        except Exception as e:
            self.log.warning("Query for application '{}' state failed with exception: '{}'.  Continuing...".
                             format(app_id, e))

        return response.data['state']

    def kill_app_by_id(self, app_id):
        """Kill an application. If the app's state is FINISHED or FAILED, it won't be changed to KILLED.
        :param app_id
        :return: The JSON response of killing the application.
        """

        response = None
        try:
            response = self.resource_mgr.cluster_application_kill(application_id=app_id)
        except Exception as e:
            self.log.warning("Termination of application '{}' failed with exception: '{}'.  Continuing...".
                             format(app_id, e))

        return response
