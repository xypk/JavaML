package mlutils.java;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.scalar.Pow;
import org.nd4j.linalg.factory.Nd4j;

public interface Evaluator {
    INDArray evaluate(INDArray pred, INDArray truth);
    String getName();

    class F1 implements Evaluator{
        public boolean holistic = false;
        public INDArray precision;
        public INDArray recall;

        public INDArray evaluate(INDArray pred, INDArray truth) {
            INDArray TV = pred.eq(truth).castTo(DataType.DOUBLE); // true values
            INDArray FV = TV.sub(1).neg(); // false values
            INDArray PV = pred.eq(Nd4j.onesLike(pred)).castTo(DataType.DOUBLE); // positive values
            INDArray TP = PV.mul(TV); // true positives
            INDArray FP = PV.mul(FV); // false positives
            INDArray FN = PV.sub(1).neg().mul(FV); // false negatives

            if(holistic) {
                TP = TP.sum();
                FP = FP.sum();
                FN = FN.sum();
            } else {
                TP = TP.sum(0);
                FP = FP.sum(0);
                FN = FN.sum(0);
            }

            this.precision = TP.div(TP.add(FP));
            this.recall = TP.div(TP.add(FN));

            return TP.mul(2).div(TP.mul(2).add(FP).add(FN));//this.precision.mul(this.recall).div(this.precision.add(this.recall)).mul(2)
        }

        public String getName() {
            return "F1 Score";
        }
    }

    class MSE implements Evaluator {
        public boolean holistic = true;

        public INDArray evaluate(INDArray pred, INDArray truth) {
            return holistic ? Nd4j.create(new double[]{pred.squaredDistance(truth)/truth.length()}) : Nd4j.exec(new Pow(pred.sub(truth), 2)).sum(0).div(pred.rows());
        }

        public String getName() {
            return "MSE";
        }
    }

    class BinaryAccuracy implements Evaluator {
        public boolean strict = false; // when enabled it only considers correct classification when ALL labels are correct and returns a scalar

        public INDArray evaluate(INDArray pred, INDArray truth) {;
            if(this.strict) {
                return pred.eq(truth).castTo(DataType.DOUBLE).sum(1).eq(truth.columns()).castTo(DataType.DOUBLE).sum().div(pred.rows());
            } else {
                return pred.eq(truth).castTo(DataType.DOUBLE).sum(0).div(pred.rows());
            }
        }

        public String getName() {
            return "Binary Accuracy";
        }
    }
}