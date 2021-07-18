package moa.classifiers.meta;

import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Instance;
import moa.capabilities.CapabilitiesHandler;
import moa.capabilities.Capability;
import moa.capabilities.ImmutableCapabilities;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.MultiClassClassifier;
import moa.core.DoubleVector;
import moa.core.Measurement;
import moa.options.ClassOption;
import moa.classifiers.trees.ARFHoeffdingTree;

public class TheEnsemble extends AbstractClassifier implements MultiClassClassifier, CapabilitiesHandler {

    private static final long serialVersionUID = 1L;

    public String getPurposeString() {
        return "A classifier that updates ensembles during every specified instance.";
    }

    public ClassOption baseLearnerOption = new ClassOption("baseLearner", 'b',
            "Classifier to train.", ARFHoeffdingTree.class, "ARFHoeffdingTree");

    public IntOption ensembleSizeOption = new IntOption("ensembleSize", 'S',
            "The number of learners.", 10, 1, Integer.MAX_VALUE);

    public IntOption windowLength = new IntOption( "limit", 'l',
            "The maximum number of instances to store", 1000, 1, Integer.MAX_VALUE);

    /**
     * The array to hold the base learners, based on the ensemble size option.
     */
    protected ARFHoeffdingTree[] ensemble;
    /**
     * The variable to hold the candidate learner model.
     */
    protected ARFHoeffdingTree candidate;
    /**
     * These three arrays are used to compute the accuracies of the ensemble.
     */
    int[] predictionCount;
    int[] correctlyClassifies;
    double[] accuracy;
    /**
     * These three variables are used to compute the accuracy of the candidate.
     */
    int candidatePredCount = 0;
    int candidateCorrect = 0;
    double candidateAcc = 0.0;
    /**
     * A counter that keeps track of window length.
     */
    int counter = 0;

    /**
     * Initalise the learners/variables.
     */
    public void resetLearningImpl() {
        // Initialise/reset all learners to begin the algorithm.
        this.ensemble = new ARFHoeffdingTree[this.ensembleSizeOption.getValue()];
        ARFHoeffdingTree treeLearner = (ARFHoeffdingTree) getPreparedClassOption(this.baseLearnerOption);
        treeLearner.resetLearning();
        // Loop through the array of ensembles.
        for (int i = 0; i < this.ensemble.length; i++) {
            // Store each learner into the ensemble array.
            this.ensemble[i] = (ARFHoeffdingTree) treeLearner.copy();
        }
        // Create a separate candidate variable to store the learner.
        candidate = (ARFHoeffdingTree) treeLearner.copy();
        // Call the method that randomises the learner hyperparameters.
        randomizeCandidate();
        // Initialise the arrays to match the size of the number of learners.
        predictionCount = new int[this.ensembleSizeOption.getValue()];
        correctlyClassifies = new int[this.ensembleSizeOption.getValue()];
        accuracy = new double[this.ensembleSizeOption.getValue()];
    }

    /**
     * Compute accuracies of the ensemble members and compares the ensemble with the candidate.
     * @param inst the incoming instance the ensembles & candidate predicts to get the accuracy.
     * @return the weighted votes.
     */
    public double[] getVotesForInstance(Instance inst) {
        // Increment counter
        counter++;
        System.out.println("Current window length: " + counter);
        // Create a vector for the combined votes.
        DoubleVector combinedVote = new DoubleVector();
        // Loop through each ensemble member with the purpose to calculate accuracy.
        for (int i = 0; i < this.ensemble.length; i++) {
            // Add to the prediction count for the current ensemble.
            predictionCount[i] += 1;
            // Check if the model has correctly classified the instance class value.
            if (this.ensemble[i].correctlyClassifies(inst)) {
                // Add to the correctly classified array.
                correctlyClassifies[i] += 1;
            }
            // Compute the accuracy array of models in the ensemble.
            accuracy[i] = (double) correctlyClassifies[i] / predictionCount[i];
            // Predict the estimated class membership probabilities for the instance.
            DoubleVector vote = new DoubleVector(this.ensemble[i].getVotesForInstance(inst));
            // Check if the votes are non-zero.
            if (vote.sumOfValues() > 0.0) {
                // Normalise the vote vector to length 1.
                vote.normalize();
                // Check if the accuracy of the model is above 0.
                if (accuracy[i] > 0.0) {
                    // Loop through each of the votes
                    for (int v = 0; v < vote.numValues(); v++) {
                        // Multiply each of the votes and put weights based on the model's accuracy.
                        vote.setValue(v, vote.getValue(v) * accuracy[i]);
                    }
                }
                combinedVote.addValues(vote);
            }
        }
        // Calculate the accuracy of the candidate model.
        candidatePredCount += 1;
        if (candidate.correctlyClassifies(inst)) {
            candidateCorrect += 1;
        }
        candidateAcc = (double) candidateCorrect / candidatePredCount;

        // Check if the counter has reached the window length.
        if (counter == this.windowLength.getValue()) {
            // Reset the counter to 0.
            counter = 0;
            // Set some variables to trace the lowest performing classifier.
            double minValue = accuracy[0];
            int minIndex = 0;
            // Pick out the ensemble model with the lowest accuracy.
            for (int i = 0; i < accuracy.length; i++) {
                if (accuracy[i] < minValue) {
                    minValue = accuracy[i];
                    minIndex = i;
                }
            }

            // Check if the candidate accuracy is greater than the minimum accuracy given by the ensemble.
            if (candidateAcc > minValue) {
                // Replace this model in the ensemble with the candidate.
                this.ensemble[minIndex] = candidate;
                // Additionally, reset the three arrays at the index that was replaced.
                predictionCount[minIndex] = 0;
                correctlyClassifies[minIndex] = 0;
                accuracy[minIndex] = 0;
            }
            // Finally, reset the candidate; randomise hyperparameters and reset its counters
            candidate.resetLearning();
            randomizeCandidate();
            candidatePredCount = 0;
            candidateCorrect = 0;
            candidateAcc = 0.0;
        }
        // Return the majority votes.
        return combinedVote.getArrayRef();
    }

    /**
     * The method used to train each ensemble member plus the candidate model.
     * @param inst the instance to be used for training
     */
    public void trainOnInstanceImpl(Instance inst) {
        // Loop through each ensemble member.
        for (int i = 0; i < this.ensemble.length; i++) {
            // Use the instances for training.
            this.ensemble[i].trainOnInstance(inst);
        }
        // Train the candidate model.
        candidate.trainOnInstance(inst);
    }

    /**
     * This method induces diversity into ARFHoeffdingTree model each time when called by randomising its hyperparameters.
     */
    protected void randomizeCandidate () {
        // Set the range of grace period values.
        int[] gracePeriod = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200};
        // Set the range of split confidence values
        double[] splitConfidence = {0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75,
        0.80, 0.85, 0.90, 0.95, 1.00};
        // Set the range of tie threshold values
        double[] tieThreshold = {0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75,
                0.80, 0.85, 0.90, 0.95, 1.00};
        // Get a random value from each of the arrays.
        int gracePeriodValue = gracePeriod[this.classifierRandom.nextInt(gracePeriod.length)];
        double splitConfidenceValue = splitConfidence[this.classifierRandom.nextInt(splitConfidence.length)];
        double tieThresholdValue = tieThreshold[this.classifierRandom.nextInt(tieThreshold.length)];
        // Set each of the randomly picked values as the option for the ARFHoeffdingTree.
        candidate.gracePeriodOption.setValue(gracePeriodValue);
        candidate.splitConfidenceOption.setValue(splitConfidenceValue);
        candidate.tieThresholdOption.setValue(tieThresholdValue);
    }

    //region Other methods.
    protected Measurement[] getModelMeasurementsImpl() {
        return new Measurement[0];
    }

    public void getModelDescription(StringBuilder out, int indent) {

    }

    public boolean isRandomizable() {
        return true;
    }

    public ImmutableCapabilities defineImmutableCapabilities() {
        if (this.getClass() == TheEnsemble.class)
            return new ImmutableCapabilities(Capability.VIEW_STANDARD, Capability.VIEW_LITE);
        else
            return new ImmutableCapabilities(Capability.VIEW_STANDARD);
    }
    //endregion
}