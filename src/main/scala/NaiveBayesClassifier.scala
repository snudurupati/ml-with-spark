/**
 * Created by snudurupati on 4/24/15.
 * A multinominal implementation of the Naive Bayes classifier.
 */
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel}
import FeatureExtraction.oneOfK

object NaiveBayesClassifier {

  case class Record(outlook: String, temperature: Int, humidity: Int, wind: String, play: Int)
  def main(args: Array[String]) {
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)

    if (args.length == 0) {
      println("Must specify training files.")
      sys.exit(1)
    }
    val conf = new SparkConf().setAppName("NaiveBayesClassifier").setMaster("local").set("spark.driver.host","localhost")
    val sc = new SparkContext(conf)
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    import sqlContext.implicits._

    //1. Data preparation
    val file = args(0)
    val data = sc.textFile(file).map(_.split(","))
      .map(p => Record(p(0), p(1).toInt, p(2).toInt, p(3), p(4).toInt)).toDF()

    val labelCol = "play"
    val featureCols = data.schema.fields.map(c => c.name).filter(c => c != labelCol)
    val categoricalCols = Array("outlook", "temperature", "humidity", "wind")

    //2. Feature extraction
    val (categoryIndexes, featuresRDD) = oneOfK(data, categoricalCols, featureCols)
    val labelsRDD = data.select(labelCol).map(r => r(0).asInstanceOf[Int].toDouble)
    val dataLP = labelsRDD.zip(featuresRDD).map(p => LabeledPoint(p._1, Vectors.dense(p._2)))

    //3. Model Fitting
    val splits = dataLP.randomSplit(Array(0.7, 0.3))
    val (trainingData, testData) = (splits(0), splits(1))
    trainingData.cache()
    println("rows in training and test data respectively %d, %d".format(trainingData.count, testData.count))
    val model = NaiveBayes.train(trainingData, 1.0)

    //4. Model Evaluation
    val labelAndpreds = testData.map(p => (p.label, model.predict(p.features)))
    val accuracy = 1.0 * labelAndpreds.filter(x => x._1 == x._2).count/testData.count

    println("The accuracy of the model is %2.4f%%".format(accuracy * 100))

  }

}
