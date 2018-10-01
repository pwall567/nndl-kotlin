/*
 * @(#) TestNNDL.kt
 *
 * nndl-kotlin Neural Networks and Deep Learning - Kotlin
 * Copyright (c) 2018 Peter Wall
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

package net.pwall.neural.test

import net.pwall.neural.Network
import net.pwall.neural.TrainingDataSource
import net.pwall.neural.TrainingDataSubset
import net.pwall.neural.test.images.MNISTImageData
import net.pwall.neural.test.images.MNISTLabelData

import java.util.Random

const val imageDataFilename = "../../Downloads/MNIST/train-images.idx3-ubyte"
const val labelDataFilename = "../../Downloads/MNIST/train-labels.idx1-ubyte"

fun main(args: Array<String>) {

    val imageData = MNISTImageData(imageDataFilename)
    val labelData = MNISTLabelData(labelDataFilename)
    val r = Random(12345)

    val network = Network(784, 30, 10)
    network.init(r)

    val tds: TrainingDataSource = InputDataSource(imageData, labelData)
    val trainingData: TrainingDataSource = TrainingDataSubset(tds, 0, 50000)
    val testData: TrainingDataSource = TrainingDataSubset(tds, 50000, 10000)

    // run stochastic gradient descent with parameters in the example

    network.stochasticGradientDescent(trainingData, 30, 10, 3.0, r, testData)

}
