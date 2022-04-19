This is a microbench on using ML framework Apache TVM for relational algebra operator.

The work is based on [1] that describes mutiple versatile workloads, e.g. SQL, graph engine processing, processed by ML framework/-s.
The assumption is that ML framework that works with tensors enables SQL, graph engines code generalisation, e.g. no need to change code for platforms with or without GPU.
Apache TVM is a good written FW that gives a tensor computation abstraction. It provides a devloper with a  low level tensor algebra computation primivitves, a so-called Tensor Expressions. 

1. http://vldb.org/pvldb/vol14/p1797-koutsoukos.pdf

