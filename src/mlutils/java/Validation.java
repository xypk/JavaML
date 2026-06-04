package mlutils.java;

import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;

import java.util.ArrayList;
import java.util.List;

public abstract class Validation {
    protected final Model model;
    protected final Evaluator eval;

    public Validation(Model model, Evaluator eval) {
        this.model = model;
        this.eval = eval;
    }
    public abstract void run();

    public static class KFold extends Validation {
        private final List<DataSet> partitions = new ArrayList<>();
        public final List<INDArray> results = new ArrayList<>();
        public double mean;

        public KFold(Model model, RecordReaderDataSetIterator iter, Evaluator eval) {
            super(model, eval);
            iter.reset();
            while (iter.hasNext()) {
                this.partitions.add(iter.next());
            }
            iter.reset();
        }

        public int getNumPartitions() {
            return this.partitions.size();
        }

        public void run() {
            System.out.printf("%d-Fold, BatchSize=%d\n", this.getNumPartitions(), this.partitions.getFirst().numExamples());

            double sum = 0;

            for (int i = 0; i < this.getNumPartitions(); i++) {
                DataSet test_set = this.partitions.get(i);
                DataSet training_set = new DataSet();

                for (int j = 0; j < this.getNumPartitions(); j++) {
                    if (j != i) {
                        training_set = DataSet.merge(java.util.Arrays.asList(training_set, this.partitions.get(j)));
                    }
                }

                this.model.reset();
                this.model.fit(training_set.getFeatures(), training_set.getLabels());
                INDArray pred = this.model.predict(test_set.getFeatures());
                INDArray result = this.eval.evaluate(pred, test_set.getLabels());
                this.results.add(result);

                sum += (double) result.meanNumber();

                    /*System.out.printf("%d. %s:", i+1, this.eval.getName());
                    for(int j = 0; j < result.size(0); j++) {
                        System.out.printf(" %.8f", result.getFloat(j));
                    }
                    System.out.println();*/
            }

            this.mean = sum / this.getNumPartitions();
            System.out.printf("Mean %s: %.4f\n", this.eval.getName(), this.mean);
        }
    }
}
