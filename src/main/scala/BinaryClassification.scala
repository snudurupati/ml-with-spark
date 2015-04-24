/**
 * Created by snudurupati on 4/16/2015.
 */

import org.apache.log4j.Logger
import org.apache.log4j.Level
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import FeatureExtraction.oneOfK

object BinaryClassification {

  case class Record(chkStatus: String, crHistory: String, presEmpSince: String, housing: String, frWorker: String, durMonths: Int, crAmt: Int, instlRate: Int, presResSince: Int, age: Int, numCredits: Int, numDeps: Int, crRating: Int)

  def main(args: Array[String]) {

    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)

    if (args.length == 0) {
      println("Must specify training files.")
      sys.exit(1)
    }

    // set up environment
    val conf = new SparkConf().setAppName("BinaryClassifier").setMaster("local").set("spark.driver.host","localhost")
    val sc = new SparkContext(conf)
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    import sqlContext.implicits._

    //Data preparation
    val file = args(0)
    val data = sc.textFile(file).map(_.split(","))
      .map(p => Record(p(0), p(1), p(4), p(8), p(11), p(12).toInt, p(13).toInt, p(14).toInt, p(15).toInt, p(16).toInt, p(17).toInt, p(18).toInt, p(19).toInt)).toDF()

    val labelCol = "credit_rating"
    val featureCols = data.schema.fields.map(c => c.name).filter(c => c != labelCol)
    val categoricalCols = Array("chkStatus", "crHistory", "presEmpSince", "housing", "frWorker")

    //Feature extraction
    val featuresRDD = oneOfK(data, categoricalCols, featureCols)
    val labelsRDD = data.select(labelCol).map(r => r(0).asInstanceOf[Int].toDouble)
    val dataLP = labelsRDD.zip(featuresRDD).map(p => LabeledPoint(p._1, Vectors.dense(p._2)))

    //Model Fitting
    val splits = dataLP.randomSplit(Array(0.7, 0.3))
    val (trainingData, testData) = (splits(0), splits(1))
    trainingData.cache()
    println("rows in training and test data respectively %d, %d".format(trainingData.count, testData.count))
    println("jsut a line added to fore commit")

    sc.stop()
  }
}
