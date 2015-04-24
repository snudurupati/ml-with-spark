/**
 * Created by snudurupati on 4/23/15.
 * Machine Learning helper library used for feature extraction, especially categorical features.
 */
import org.apache.spark.sql.{DataFrame, Row}

object FeatureExtraction {

  /*
  Performs 1-of-k encoding on categorical columns and returns an RDD of the feature array
   */
  def oneOfK(data: DataFrame, categoricalCols: Array[String], featureCols: Array[String]) = {

    //takes a categorical column, performs 1-of-k encoding and returns a Mao of category -> index
    def indexColumn(col: String) = {
      data.select(col).map(r => r(0)).distinct.collect.zipWithIndex.toMap
    }

    //encodes each categorical column as a Map of category -> index and return a Map of Maps
    val categoryIndexes = categoricalCols.map(c => (c, indexColumn(c))).toMap

    //replaces categorical features with corresponding indexes and returns a features array
    def reIndexRow(row: Row) = {
      val features = Array.ofDim[Double](row.size)
      for (i <- 0 to row.size - 2) {
        if (categoricalCols.contains(featureCols(i)))
          features(i) = categoryIndexes(featureCols(i))(row(i))
        else
          features(i) = row(i).asInstanceOf[Int].toDouble
      }
      features
    }

    val featureRDD = data.map(r => reIndexRow(r))

    (categoryIndexes, featureRDD)
  }

}
