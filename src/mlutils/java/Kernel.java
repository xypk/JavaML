package mlutils.java;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.scalar.Pow;
import org.nd4j.linalg.api.ops.impl.transforms.strict.Exp;
import org.nd4j.linalg.factory.Nd4j;

public interface Kernel {
    INDArray out(INDArray in);

    class GaussianKernel implements Kernel {
        private static final double inv_sqrt2pi = 1/Math.sqrt(2*Math.PI);

        public INDArray out(INDArray in) {
            return Nd4j.exec(new Exp(Nd4j.exec(new Pow(in,2)).div(2).neg())).mul(inv_sqrt2pi);
        }
    }
}