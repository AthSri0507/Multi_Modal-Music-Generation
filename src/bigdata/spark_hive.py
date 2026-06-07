"""Hive-enabled SparkSession factory for the big-data warehouse.

Provides a single helper, :func:`build_hive_spark`, that creates a SparkSession
with ``enableHiveSupport()`` backed by an embedded Derby metastore and a warehouse
directory. The warehouse lives on the local filesystem by default but can be
pointed at HDFS by setting the ``HDFS_BASE_URI`` env var / config -- no code change
needed to move to the Docker Hadoop+Hive cluster.

Real HiveQL (``CREATE DATABASE``, managed ``saveAsTable`` tables, ``spark.sql``
queries) runs through this session, so the big-data layer is genuinely Hive-based
even on a single Windows/CPU machine.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

from pyspark.sql import SparkSession

from src.config.config import (
    HDFS_BASE_URI,
    HIVE_DB_NAME,
    HIVE_METASTORE_DIR,
    HIVE_WAREHOUSE_DIR,
    PROJECT_ROOT,
    SPARK_DRIVER_MEMORY,
    SPARK_EXECUTOR_MEMORY,
    SPARK_MASTER,
)


def _ensure_hadoop_home() -> None:
    """Point HADOOP_HOME at the project-local winutils on Windows.

    Spark's Hive metastore needs winutils.exe + hadoop.dll on native Windows
    (see scripts/setup_winutils or the bundled ``hadoop/`` dir). If HADOOP_HOME is
    already set we respect it; otherwise we use the project-local copy when present.
    """
    if os.name != "nt" or os.environ.get("HADOOP_HOME"):
        return
    hadoop_home = Path(PROJECT_ROOT) / "hadoop"
    winutils = hadoop_home / "bin" / "winutils.exe"
    if winutils.exists():
        os.environ["HADOOP_HOME"] = str(hadoop_home)
        os.environ["hadoop.home.dir"] = str(hadoop_home)
        bin_dir = str(hadoop_home / "bin")
        if bin_dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")


def _warehouse_location(hdfs_base: Optional[str]) -> str:
    """Resolve the Hive warehouse location (HDFS when configured, else local).

    On local FS we return a plain forward-slash path rather than a ``file://`` URI:
    ``Path.as_uri()`` percent-encodes spaces (e.g. ``Atharva%20Srivastava``), which
    Hadoop/Hive then fails to create on Windows. A literal-space path works.
    """
    if hdfs_base:
        return f"{hdfs_base.rstrip('/')}/hive_warehouse"
    Path(HIVE_WAREHOUSE_DIR).mkdir(parents=True, exist_ok=True)
    return str(Path(HIVE_WAREHOUSE_DIR).resolve()).replace("\\", "/")


def build_hive_spark(
    app_name: str = "music_gen_warehouse",
    master: Optional[str] = None,
    hdfs_base: Optional[str] = HDFS_BASE_URI,
    create_db: bool = True,
) -> SparkSession:
    """Create (or reuse) a Hive-enabled SparkSession.

    Parameters
    ----------
    app_name:
        Spark application name.
    master:
        Spark master URL; defaults to ``config.SPARK_MASTER`` (``local[*]``).
    hdfs_base:
        Optional HDFS base URI; when provided the warehouse is stored on HDFS.
    create_db:
        When True, ensures the ``music`` database exists.
    """
    # Windows / single-node friendly networking + interpreter pinning.
    _ensure_hadoop_home()
    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
    os.environ.setdefault("SPARK_DRIVER_HOST", "127.0.0.1")
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    Path(HIVE_METASTORE_DIR).parent.mkdir(parents=True, exist_ok=True)
    warehouse = _warehouse_location(hdfs_base)
    # Embedded Derby metastore rooted at HIVE_METASTORE_DIR. Forward slashes avoid
    # backslash escaping issues in the JDBC URL on Windows. Hive metastore config
    # must be prefixed with ``spark.hadoop.`` or Spark ignores it.
    derby_path = str(Path(HIVE_METASTORE_DIR).resolve()).replace("\\", "/")
    derby_url = f"jdbc:derby:;databaseName={derby_path};create=true"

    builder = (
        SparkSession.builder.appName(app_name)
        .master(master or SPARK_MASTER)
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.pyspark.python", sys.executable)
        .config("spark.pyspark.driver.python", sys.executable)
        .config("spark.driver.memory", SPARK_DRIVER_MEMORY)
        .config("spark.executor.memory", SPARK_EXECUTOR_MEMORY)
        .config("spark.sql.warehouse.dir", warehouse)
        .config("spark.hadoop.javax.jdo.option.ConnectionURL", derby_url)
        # Quieter single-node startup.
        .config("spark.ui.showConsoleProgress", "false")
        .enableHiveSupport()
    )

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    if create_db:
        spark.sql(f"CREATE DATABASE IF NOT EXISTS {HIVE_DB_NAME}")
        spark.sql(f"USE {HIVE_DB_NAME}")
    return spark


def table_fqn(table: str) -> str:
    """Fully-qualified Hive table name (``music.<table>``)."""
    return f"{HIVE_DB_NAME}.{table}"
