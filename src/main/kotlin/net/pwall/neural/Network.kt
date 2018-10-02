/*
 * @(#) Network.kt
 *
 * nndl-kotlin Neural Networks and Deep Learning - Kotlin
 * Copyright (c) 2018 Peter Wall
 * Derived from original Python code copyright (c) 2012-2015 Michael Nielsen
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

package net.pwall.neural

import java.util.Random

/**
 * Neural Network.  This implementation is based on the work of Michael Nielsen in the online
 * book [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/).
 *
 * I cannot speak highly enough of this book, which gives a clear explanation of the theory
 * and mathematics of neural networks as well as implementation code in Python which formed the
 * basis of this Kotlin version.
 *
 * @author      Peter Wall
 * @author      Michael Nielsen (original Python code)
 */
class Network(vararg layerSizes: Int) {

    val numLayers = layerSizes.size
    private val numHiddens = numLayers - 1

    init {
        if (numLayers < 2)
            throw IllegalArgumentException("Must have 2 or more layers")
    }

    private val inputLayer = InputLayer(layerSizes[0])
    private val hiddenLayers = initHiddenLayers(*layerSizes)

    private fun initHiddenLayers(vararg layerSizes: Int) : Array<HiddenLayer> {
        val list = mutableListOf(HiddenLayer(inputLayer, layerSizes[1]))
        for (i in 1 until numHiddens)
            list.add(HiddenLayer(list[i - 1], layerSizes[i + 1]))
        return list.toTypedArray()
    }

    /**
     * Set the values of all the inputs in a single operation.
     *
     * @param   inputs   the new values
     */
    fun setInputs(inputs: DoubleArray) {
        inputLayer.setInputs(inputs)
    }

    /**
     * Get the output layer (in this implementation all layers other than the input layer are
     * called hidden layers; the output layer is the last hidden layer).
     *
     * @return  the output layer
     */
    fun getOutputLayer() : HiddenLayer = hiddenLayers.last()

    /**
     * Get the outputs of the output layer.
     *
     * @return  the outputs
     */
    fun getOutputs() : DoubleArray = getOutputLayer().outputs

    /**
     * Initialise the network using the supplied [Random].  The [Random] may be supplied
     * as an argument to allow the user to use a [Random] with a known seed for repeatable
     * results.
     *
     * @param   r   the [Random]
     */
    fun init(r: Random = Random()) {
        for (hiddenLayer in hiddenLayers)
            hiddenLayer.init(r)
    }

    /**
     * Process an array of inputs to produce an array of outputs.  This is the principal
     * function of a neural network, but for most purposes the full array of outputs is not
     * required, just the index of the highest output.  For this usage, see [getResultInt].
     *
     * @param   inputs  the array of inputs (no length checking is performed)
     * @return
     */
    fun getResultArray(inputs: DoubleArray) : DoubleArray {
        setInputs(inputs)
        for (hiddenLayer in hiddenLayers)
            hiddenLayer.iterate()
        return getOutputs()
    }

    /**
     * Process an array of inputs to get a single integer output - the index of the highest
     * value in the output array.
     *
     * @param   inputs  the array of inputs (no length checking is performed)
     * @return  the index of the highest output
     */
    fun getResultInt(inputs: DoubleArray) = indexOfHighest(getResultArray(inputs))

    /**
     * Implementation of the mini-batch Stochastic Gradient Descent algorithm.
     *
     * See [Neural Networks and Deep Learning, Chapter 1](http://neuralnetworksanddeeplearning.com/chap1.html) for a
     * full description of this functionality.
     *
     * @param   tds             a [TrainingDataSource]
     * @param   epochs          number of epochs
     * @param   miniBatchSize   the size of a mini-batch
     * @param   eta             the learning rate
     * @param   r               a [Random], used to shuffle the training data (see the note on [init])
     * @param   testData        a second [TrainingDataSource] containing test data to
     *                          evaluate progress (may be `null`)
     * @throws  IllegalArgumentException if the number of epochs not in allowed range
     */
    fun stochasticGradientDescent(tds: TrainingDataSource, epochs: Int, miniBatchSize: Int, eta: Double,
            r: Random = Random(), testData: TrainingDataSource?) {
        println("Stochastic Gradient Descent on ${toString()}; training data ${tds.getSize()}; $epochs epochs; " +
                "mini-batch size $miniBatchSize; eta $eta")
        val tdr = TrainingDataRandom(tds)
        if (epochs !in 1..200)
            throw IllegalArgumentException("number of epochs must be in range 1..200")
        var startTime = System.currentTimeMillis()
        for (epoch in 1..epochs) {
            tdr.randomise(r)
            var k = 0
            while (k < tdr.getSize()) {
                val miniBatch = TrainingDataSubset(tdr, k, Math.min(miniBatchSize, tdr.getSize() - k))
                updateMiniBatch(miniBatch, eta)
                k += miniBatchSize
            }
            var now = System.currentTimeMillis()
            println("Completed epoch $epoch (${now - startTime}ms)")
            startTime = now
            if (testData != null) {
                val n = evaluate(testData)
                now = System.currentTimeMillis()
                println("Correctly identified $n of ${testData.getSize()} (${now - startTime}ms)")
                startTime = now
            }
        }
    }

    /**
     * Implementation of "update_mini_batch".
     *
     * @param   miniBatch       the mini-batch
     * @param   eta             the learning rate
     */
    private fun updateMiniBatch(miniBatch: TrainingDataSubset, eta: Double) {
        val nablaB = Array(numHiddens) { i -> hiddenLayers[i].getZeroBiasesArray() }
        val nablaW = Array(numHiddens) { i -> hiddenLayers[i].getZeroWeightsArray() }

        for (m in 0 until miniBatch.getSize()) {
            val (deltaNablaB, deltaNablaW) = backProp(miniBatch.getItem(m))
            for (i in 0 until numHiddens) {
                nablaB[i].plusAssign(deltaNablaB[i])
                nablaW[i].plusAssign(deltaNablaW[i])
            }
        }

        for (i in 0 until numHiddens) {
            val h = hiddenLayers[i]
            h.weights -= nablaW[i] * (eta / miniBatch.getSize())
            h.biases -= nablaB[i] * (eta / miniBatch.getSize())
        }
    }

    /**
     * Calculate gradient for the cost function.  `nablaB` and `nablaW` are layer-by-layer lists of [NDArray], similar
     * to `biases` and `weights`.
     *
     * @param   td      the training data item
     * @return          a [Pair] (`nablaB`, `nablaW`) representing the gradient for the cost function
     */
    private fun backProp(td: TrainingData) : Pair<Array<NDArray>, Array<NDArray>> {

        val x = NDArray.fromDoubleArray(td.getInputs())
        val y = NDArray.fromDoubleArray(td.getOutputs())

        val nablaB = Array(numHiddens) { i -> hiddenLayers[i].getZeroBiasesArray() }
        val nablaW = Array(numHiddens) { i -> hiddenLayers[i].getZeroWeightsArray() }

        // feedforward
        var activation = x
        val activations = mutableListOf(activation)
        val zs = mutableListOf<NDArray>()
        for (i in hiddenLayers.indices) {
            val h = hiddenLayers[i]
            val z = (h.weights dot activation) + h.biases
            zs.add(z)
            activation = z.apply(Companion::sigmoid)
            activations.add(activation)
        }

        // backward pass
        var delta = costDerivative(activations.last(), y) * zs.last().apply(Companion::sigmoidPrime)
        nablaB[nablaB.size - 1] = delta
        nablaW[nablaW.size - 1] = delta dot activations[activations.size - 2].transpose()

        for (l in 2 until numLayers) {
            val z = zs[zs.size - l]
            val sp = z.apply(Companion::sigmoidPrime)
            delta = (hiddenLayers[numHiddens - l + 1].weights.transpose() dot delta) * sp
            nablaB[numHiddens - l] = delta
            nablaW[numHiddens - l] = delta dot activations[activations.size - l - 1].transpose()
        }

        return Pair(nablaB, nablaW)
    }

    private fun costDerivative(outputActivations: NDArray, y: NDArray) : NDArray {
        return outputActivations - y
    }

    /**
     * Return the number of test inputs for which the neural network outputs the correct result.
     * Note that the neural network's output is assumed to be the index of whichever neuron in
     * the final layer has the highest activation.
     *
     * @param   testData    the set of test data
     * @return              the total number of correct results
     */
    fun evaluate(testData: TrainingDataSource): Int {
        var sum = 0
        for (i in 0 until testData.getSize()) {
            val td = testData.getItem(i)
            if (getResultInt(td.getInputs()) == td.getHighestOutputIndex())
                sum++
        }
        return sum
    }

    override fun toString(): String {
        val sb = StringBuilder("Network[")
        sb.append(inputLayer.size)
        for (h in hiddenLayers)
            sb.append(',').append(h.size)
        sb.append(']')
        return sb.toString()
    }

    companion object {

        fun sigmoid(z: Double) = 1.0 / (1.0 + Math.exp(-z))

        fun sigmoidPrime(z: Double) : Double {
            val s = sigmoid(z)
            return s * (1.0 - s)
        }

        fun indexOfHighest(a: DoubleArray) : Int {
            var result = 0
            var highest = a[0]
            for (i in 1 until a.size) {
                if (a[i] > highest) {
                    highest = a[i]
                    result = i
                }
            }
            return result
        }

    }

}
