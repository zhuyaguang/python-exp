
* helm repo add radondb https://radondb.github.io/radondb-mysql-kubernetes/


* helm install demo radondb/mysql-operator

* kubectl apply -f mysql_v1alpha1_mysqlcluster.yaml

* 打开 mysql 监控

  metricsOpts:
    enabled: false