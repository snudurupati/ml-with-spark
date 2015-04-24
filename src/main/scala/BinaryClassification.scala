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
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
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

    //1. Data preparation
    val file = args(0)
    val data = sc.textFile(file).map(_.split(","))
      .map(p => Record(p(0), p(1), p(4), p(8), p(11), p(12).toInt, p(13).toInt, p(14).toInt, p(15).toInt, p(16).toInt, p(17).toInt, p(18).toInt, p(19).toInt)).toDF()

    val labelCol = "crRating"
    val featureCols = data.schema.fields.map(c => c.name).filter(c => c != labelCol)
    val categoricalCols = Array("chkStatus", "crHistory", "presEmpSince", "housing", "frWorker")

    //2. Feature extraction
    val (categoryIndexes, featuresRDD) = oneOfK(data, categoricalCols, featureCols)
    val labelsRDD = data.select(labelCol).map(r => r(0).asInstanceOf[Int].toDouble)
    val dataLP = labelsRDD.zip(featuresRDD).map(p => LabeledPoint(p._1, Vectors.dense(p._2)))

    //3. Model Fitting
    val splits = dataLP.randomSplit(Array(0.7, 0.3))
    val (trainingData, testData) = (splits(0), splits(1))
    trainingData.cache()
    println("rows in training and test data respectively %d, %d".format(trainingData.count, testData.count))

    //Configure the decision tree and train the model

    //Construct a Map of categorical features
    def getCategoryFeatureInfo() = {
      val featureColsMap = featureCols.zipWithIndex.toMap
      val catInfoMap = categoryIndexes.map{ case(k, v) =>
        (featureColsMap(k), v.size)
      }
      catInfoMap
    }

    val numClasses = 2
    val categoricalFeaturesInfo = getCategoryFeatureInfo()
    val impurity = "gini"
    val maxDepth = 4
    val maxBins = 32

    val model = DecisionTree.trainClassifier(trainingData, numClasses, categoricalFeaturesInfo, impurity, maxDepth, maxBins)
    println("Learned classification tree model:\n%s".format(model.toDebugString))

    //4. Model Evaluation
    val labelAndPreds = testData.map{ p =>
      val prediction = model.predict(p.features)
      (p.label, prediction) }
    val dtAccuracy = labelAndPreds.filter(r => r._1 == r._2).count.toDouble/testData.count()
    //val testErr = labelAndPreds.filter(r => r._1 != r._2).count.toDouble/testData.count()

    println("The accuracy of the model is %2.2f%%\n".format(dtAccuracy*100))

    val metrics = Seq(model).map{ model =>
      val labelsAndpreds = testData.map { point =>
        val score = model.predict(point.features)
        (if (score > 0.5) 1.0 else 0.0, point.label)
      }
      val metrics = new BinaryClassificationMetrics(labelsAndpreds)
      (model.getClass.getSimpleName, metrics.areaUnderPR,metrics.areaUnderROC)
    }

    metrics.foreach { case (m, pr, roc) =>
      println("Area under PR: %2.4f%%\nArea under ROC: %2.4f%%\n".format(pr * 100.0, roc * 100.0))
      //println(f"$m, Area under PR: ${pr * 100.0}%2.4f%%, Area under ROC: ${roc * 100.0}%2.4f%%")
    }
    sc.stop()
  }
}
