/**
 * Created by snudurupati on 4/16/2015.
 */
package com.snudurupati.apps

import org.apache.log4j.Logger
import org.apache.log4j.Level
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf

object binaryClassification {
  def main(args: Array[String]) {

    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)

    if (args.length == 0) {
      println("Must specify training files.")
      sys.exit(1)
    }

    // set up environment
    val conf = new SparkConf().setAppName("BinaryClassifier")
    val sc = new SparkContext(conf)

    //Code
    val file1 = args(0)
    val file2 = args(1)
    val file3 = args(2)

    val raw_data = sc.textFile(file1)
    val credit_values = sc.textFile(file2)
    val checking_status = sc.textFile(file3)



    sc.stop()
  }
}
