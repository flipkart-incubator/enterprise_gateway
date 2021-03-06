This image enables the use of an IRKernel kernel launched from [Jupyter Enterprise Gateway](http://jupyter-enterprise-gateway.readthedocs.io/en/latest/) within a Kubernetes cluster.  It is currently built on [kubespark/spark-driver-r:v2.2.0-kubernetes-0.5.0](https://hub.docker.com/r/kubespark/spark-driver-r/) deriving from the [apache-spark-on-k8s](https://github.com/apache-spark-on-k8s/spark) fork.

# What it Gives You
* IRKernel kernel support 
* Spark on kubernetes support from within a Jupyter Notebook

# Basic Use
Pull [elyra/enterprise-gateway](https://hub.docker.com/r/elyra/enterprise-gateway/), along with all of the elyra/kernel-* images to each of your kubernetes nodes.  Although manual seeding of images across the cluster is not required, it is highly recommended since kernel startup times can timeout and image downloads can seriously undermine that window.

Download the [enterprise-gateway.yaml](https://github.com/jupyter-incubator/enterprise_gateway/blob/master/etc/kubernetes/enterprise-gateway.yaml) file and make any necessary changes for your configuration.  We recommend that a persistent volume be used so that the kernelspec files can be accessed outside of the container since we've found those to require post-deployment modifications from time to time.

Deploy Jupyter Enterprise Gateway using `kubectl apply -f enterprise-gateway.yaml`

Launch a Jupyter Notebook application using NB2KG (see [elyra/nb2kg](https://hub.docker.com/r/elyra/nb2kg/) against the Enterprise Gateway instance and pick either of the python-related kernels.

For more information, check our [repo](https://github.com/jupyter-incubator/enterprise_gateway) and [docs](http://jupyter-enterprise-gateway.readthedocs.io/en/latest/).
