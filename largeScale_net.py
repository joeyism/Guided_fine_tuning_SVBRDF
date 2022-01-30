from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import cv2
import imageio
import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import argparse
import json
import glob
import random
import collections
import math
import time
import dataReader
import model as mod
import losses
import helpers
import shutil
from random import shuffle
inputSize = 512
strideSize = 256
exampleSize = 256

#!!!If running TF v > 2.0 uncomment those lines (also remove the tensorflow import on line 5):!!!
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()

#Under MIT License

#Source code tested for tensorflow version 1.12

a = type('', (), {})()
a.useAmbientLight = False
a.NoAugmentationInRenderings = True
a.logOutputAlbedos = False
a.max_epochs = 10

TILE_SIZE = 512
inputpythonList = []

def main():
    a.seed = random.randint(0, 2**31 - 1)

    tf.set_random_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)
    loadCheckpointOption(a.mode, a.checkpoint) #loads so that I don't mix up options and it generates data corresponding to this training

    config = tf.ConfigProto()

    if not os.path.exists(a.output_dir):
        os.makedirs(a.output_dir)

    with open(os.path.join(a.output_dir, "options.json"), "w") as f:
        f.write(json.dumps(vars(a), sort_keys=True, indent=4))

    data = dataReader.dataset(a.input_dir, imageFormat = a.imageFormat, trainFolder = a.trainFolder, testFolder = a.testFolder, nbTargetsToRead = a.nbTargets, tileSize=TILE_SIZE, inputImageSize=a.input_size, batchSize=a.batch_size, fixCrop = (a.mode == "test"), mixMaterials = (a.mode == "train" or a.mode == "finetune"), logInput = a.useLog, useAmbientLight = a.useAmbientLight, useAugmentationInRenderings = not a.NoAugmentationInRenderings)
    # Populate data
    data.loadPathList(a.inputMode, a.mode, a.mode == "train" or a.mode == "finetune", inputpythonList)

    if a.feedMethod == "render":
        if a.mode == "train":
            data.populateInNetworkFeedGraph(a.renderingScene, a.jitterLightPos, a.jitterViewPos,  shuffle = (a.mode == "train"  or a.mode == "finetune"))
        elif a.mode == "finetune":
            data.populateInNetworkFeedGraphSpatialMix(a.renderingScene, shuffle = False, imageSize = a.input_size)

    elif a.feedMethod == "files":
        data.populateFeedGraph(shuffle = (a.mode == "train"  or a.mode == "finetune"))


    if a.mode == "train" or a.mode == "finetune":
        with tf.name_scope("recurrentTest"):
            dataTest = dataReader.dataset(a.input_dir, imageFormat = a.imageFormat, testFolder = a.testFolder, nbTargetsToRead = a.nbTargets, tileSize=TILE_SIZE, inputImageSize=a.test_input_size, batchSize=a.batch_size, fixCrop = True, mixMaterials = False, logInput = a.useLog, useAmbientLight = a.useAmbientLight, useAugmentationInRenderings = not a.NoAugmentationInRenderings)
            dataTest.loadPathList(a.inputMode, "test", False, inputpythonList)
            if a.testApproach == "render":
                #dataTest.populateInNetworkFeedGraphSpatialMix(a.renderingScene, shuffle = False, imageSize = TILE_SIZE, useSpatialMix=False)
                dataTest.populateInNetworkFeedGraph(a.renderingScene, a.jitterLightPos, a.jitterViewPos, shuffle = False)
            elif a.testApproach == "files":
                dataTest.populateFeedGraph(False) 

    targetsReshaped = helpers.target_reshape(data.targetBatch)

    #CreateModel
    model = mod.Model(data.inputBatch, generatorOutputChannels=9)
    model.create_model()
    if a.mode == "train" or a.mode == "finetune":
        testTargetsReshaped = helpers.target_reshape(dataTest.targetBatch)

        testmodel = mod.Model(dataTest.inputBatch, generatorOutputChannels=9, reuse_bool=True)

        testmodel.create_model()
        display_fetches_test, _ = helpers.display_images_fetches(dataTest.pathBatch, dataTest.inputBatch, dataTest.targetBatch, dataTest.gammaCorrectedInputsBatch, testmodel.output, a.nbTargets, a.logOutputAlbedos)

        loss = losses.Loss(a.loss, model.output, targetsReshaped, TILE_SIZE, a.batch_size, tf.placeholder(tf.float64, shape=(), name="lr"), a.includeDiffuse, a.nbSpecularRendering, a.nbDiffuseRendering)

        loss.createLossGraph()
        loss.createTrainVariablesGraph()

    #Register Renderings And Loss In Tensorflow
    display_fetches, converted_images = helpers.display_images_fetches(data.pathBatch, data.inputBatch, data.targetBatch, data.gammaCorrectedInputsBatch, model.output, a.nbTargets, a.logOutputAlbedos)
    if a.mode == "train":
        helpers.registerTensorboard(data.pathBatch, converted_images, a.nbTargets, loss.lossValue, a.batch_size, loss.targetsRenderings, loss.outputsRenderings)

    #Run either training or test
    with tf.name_scope("parameter_count"):
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])
    saver = tf.train.Saver(max_to_keep=1)
    
    if a.checkpoint is not None:
        print("reading model from checkpoint : " + a.checkpoint)
        checkpoint = tf.train.latest_checkpoint(a.checkpoint)
        partialSaver = helpers.optimistic_saver(checkpoint) #Be careful this will silently not load variables if they are missing from the graph or checkpoint
        
    logdir = a.output_dir if a.summary_freq > 0 else None
    sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=0, saver=None)

    with sv.managed_session("", config= config) as sess:
        sess.run(data.iterator.initializer)
        print("parameter_count =", sess.run(parameter_count))

        if a.checkpoint is not None:
            print("restoring model from checkpoint : " + a.checkpoint)
            partialSaver.restore(sess, checkpoint)

        max_steps = 2**32
        if a.max_epochs is not None:
            max_steps = data.stepsPerEpoch * a.max_epochs
        if a.max_steps is not None:
            max_steps = a.max_steps

        sess.run(data.iterator.initializer)
        if a.mode == "test":
            filesets = test(sess, data, max_steps, display_fetches, output_dir = a.output_dir)

        if a.mode == "train"  or a.mode == "finetune":
           train(sv, sess, data, max_steps, display_fetches, display_fetches_test, dataTest, saver, loss, a.output_dir)



def loadCheckpointOption(mode, checkpoint):
    if mode == "test":
        if checkpoint is None:
            raise Exception("checkpoint required for test mode")
    
    if not checkpoint is None:
        # load some options from the checkpoint
        options = {"which_direction", "ngf", "nbTargets", "loss", "useLog", "includeDiffuse"}
        with open(os.path.join(checkpoint, "options.json")) as f:
            for key, val in json.loads(f.read()).items():
                if key in options:
                    print("loaded", key, "=", val)
                    setattr(a, key, val)

    for k, v in a.__dict__.items():
        print(k, "=", v)

def test(sess, data, max_steps, display_fetches, output_dir):
    #testing at most, process the test data once
    sess.run(data.iterator.initializer)
    max_steps = min(data.stepsPerEpoch, max_steps)
    filesets = []
    for step in range(max_steps):
        try:
            results = sess.run(display_fetches)
            filesets.extend(helpers.save_images(results, output_dir, a.batch_size, a.nbTargets))

        except tf.errors.OutOfRangeError:
            print("testing fails in OutOfRangeError")
            continue
    index_path = helpers.append_index(filesets, output_dir, a.nbTargets, a.mode)
    return filesets

def train(sv, sess, data, max_steps, display_fetches, display_fetches_test, dataTest, saver, loss, output_dir):
    sess.run(data.iterator.initializer)
    try:
        # training
        start_time = time.time()

        for step in range(max_steps):
            options = None
            run_metadata = None
            if helpers.should(a.trace_freq, max_steps, step):
                options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()

            fetches = {
                "train": loss.trainOp,
                "global_step": sv.global_step,
            }

            if helpers.should(a.progress_freq, max_steps, step) or step <= 1:
                fetches["loss_value"] = loss.lossValue

            if helpers.should(a.summary_freq, max_steps, step):
                fetches["summary"] = sv.summary_op

            fetches["display"] = display_fetches
            try:
                currentLrValue = a.lr
                if a.checkpoint is None and step < 500:
                    currentLrValue = step * (0.002) * a.lr # ramps up to a.lr in the 2000 first iterations to avoid crazy first gradients to have too much impact.

                results = sess.run(fetches, feed_dict={loss.lr: currentLrValue}, options=options, run_metadata=run_metadata)
            except tf.errors.OutOfRangeError :
                print("training fails in OutOfRangeError, probably a problem with the iterator")
                continue

            global_step = results["global_step"]
            
            #helpers.saveInputs(a.output_dir, results["display"], step)

            if helpers.should(a.summary_freq, max_steps, step):
                sv.summary_writer.add_summary(results["summary"], global_step)

            if helpers.should(a.trace_freq, max_steps, step):
                print("recording trace")
                sv.summary_writer.add_run_metadata(run_metadata, "step_%d" % global_step)

            if helpers.should(a.progress_freq, max_steps, step):
                # global_step will have the correct step count if we resume from a checkpoint
                train_epoch = math.ceil(global_step / data.stepsPerEpoch)
                train_step = global_step - (train_epoch - 1) * data.stepsPerEpoch
                imagesPerSecond = global_step * a.batch_size / (time.time() - start_time)
                remainingMinutes = ((max_steps - global_step) * a.batch_size)/(imagesPerSecond * 60)
                print("progress  epoch %d  step %d  image/sec %0.1f" % (train_epoch, global_step, imagesPerSecond))
                print("Remaining %0.1f minutes" % (remainingMinutes))
                print("loss_value", results["loss_value"])

            if helpers.should(a.save_freq, max_steps, step):
                print("saving model")
                try:
                    saver.save(sess, os.path.join(output_dir, "model"), global_step=sv.global_step)
                except Exception as e:
                    print("Didn't manage to save model (trainining continues): " + str(e))

            if helpers.should(a.test_freq, max_steps, step) or global_step == 1:
                outputTestDir = os.path.join(a.output_dir, str(global_step))
                try:
                    test(sess, dataTest, max_steps, display_fetches_test, outputTestDir)
                except Exception as e:
                    print("Didn't manage to do a recurrent test (trainining continues): " + str(e))

            if sv.should_stop():
                break
    finally:
        saver.save(sess, os.path.join(output_dir, "model"), global_step=sv.global_step) #Does the saver saves everything still ?
        sess.run(data.iterator.initializer)
        outputTestDir = os.path.join(a.output_dir, "final")
        test(sess, dataTest, max_steps, display_fetches_test, outputTestDir )

def runNetwork(inputDir, outputDir, checkpoint, inputMode = "image", feedMethod = "files", mode="test", input_size=512, nbTargets = 1, batch_size = 1, fileList = [], nbStepMax = 3000, testApproach = "render"):
    inputpythonList.clear()
    a.inputMode = inputMode
    a.feedMethod = feedMethod
    a.input_dir = inputDir
    a.output_dir = outputDir
    a.checkpoint = checkpoint
    a.mode = mode
    a.input_size = input_size
    a.nbTargets = nbTargets
    a.batch_size = batch_size
    a.renderingScene = "diffuse"
    a.max_steps = nbStepMax
    a.save_freq = 100000
    a.test_freq = 1500
    a.progress_freq = 500
    a.loss = "mixed"
    a.lr = 0.00002
    a.useLog = True
    a.summary_freq = 100000
    a.jitterLightPos = True
    a.jitterViewPos = True
    a.includeDiffuse = True
    a.testApproach = testApproach
    a.test_input_size = 512
    a.imageFormat = "png"
    a.trainFolder = "train"
    a.testFolder = "test"
    
    inputpythonList.extend(fileList)
    tf.reset_default_graph()
    print(a)
    #setup all options...
    main()

def cropImage(imagePath, materialName, output_dir):
    output_test_dir = os.path.join(output_dir, "test")
    if not os.path.exists(output_test_dir):
        os.makedirs(output_test_dir)

    currentImageName = materialName
    image = imageio.imread(imagePath)
    height = int(image.shape[0])
    width = int(image.shape[1])
    height = int(image.shape[0]/inputSize)*inputSize
    width = int(image.shape[1]/inputSize)*inputSize
    image = image[:height, :width, :]
    if image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    cv2.imwrite(imagePath, image)

    widthSplit = int(np.ceil(width / strideSize)) - 1
    heightSplit = int(np.ceil(height / strideSize)) - 1

    #Split the image
    maxIDImage = 0

    for i in range(widthSplit):
        for j in range(heightSplit):
            currentJPix = j * strideSize
            currentIPix = i * strideSize
            split = image[currentJPix:currentJPix + inputSize, currentIPix:currentIPix + inputSize, :]
            splitID = (i * heightSplit) + j

            currentSplitPath = os.path.join(output_test_dir, currentImageName + "_" + str(splitID) + ".png")

            im = imageio.imwrite(currentSplitPath, split)

            maxIDImage = splitID
    return maxIDImage, widthSplit, heightSplit, height, width

def stitchResults(inputSize, outputFolder, maxIDImage, networkOutputsFolder, materialName, height, width, widthSplit, heightSplit):
    #We define here the weight to apply to each tiles, with one in the center and 0 in the borders.
    sigm = 0.20
    maxVal = gaussianWeight(0.5, 0.5, sigm)
    oneImageWeights = np.asarray(np.meshgrid(np.linspace(0, 1, inputSize), np.linspace(0, 1, inputSize)))
    oneImageWeights = gaussianWeight(oneImageWeights, 0.5, sigm) / maxVal
    oneImageWeights = oneImageWeights[0] * oneImageWeights[1]
    oneImageWeights = np.expand_dims(oneImageWeights, axis=-1)

    #Which folder is going to hold the stitched final results.
    folderOutput = os.path.join(outputFolder, "results_fineTuned")
    if not os.path.exists(folderOutput):
        os.makedirs(folderOutput)

    #for each map id (representing normal, diffuse, roughness and specular)
    for idMap in range(4):
        allImages = []
        #We store in memory all the results for the different tiles for the current map type.
        for idImage in range(maxIDImage + 1):
            imagePath = os.path.join(networkOutputsFolder,"final", "images", materialName + "_" + str(idImage)+"-outputs_gammad-" + str(idMap) + "-.png" )
            allImages.append(imageio.imread(imagePath))

        #We initialize the final image and the weights we will use to normalize the contribution of each tile
        finalImage = np.zeros((height, width, 3))
        finalWeights = np.zeros((height, width, 3))
        for i in range(widthSplit):
            for j in range(heightSplit):
                currentJPix = j * strideSize
                currentIPix = i * strideSize
                splitID = (i * heightSplit) + j
                #We now paste each images weight by the gaussian weights in the final image at the proper position
                finalImage[currentJPix:currentJPix + inputSize, currentIPix:currentIPix + inputSize, :] = finalImage[currentJPix:currentJPix + inputSize, currentIPix:currentIPix + inputSize, :] + ((allImages[splitID]/255.0) * oneImageWeights)
                #And creates a final weight image that stores the different total weights applied to each pixel, to normalize it in the end
                finalWeights[currentJPix:currentJPix + inputSize, currentIPix:currentIPix + inputSize, :] = finalWeights[currentJPix:currentJPix + inputSize, currentIPix:currentIPix + inputSize, :] +oneImageWeights
        #Normalizes the image with respect to each pixel's weight.
        finalImage = finalImage / finalWeights
        #Saves the map as uint8.
        finalImage = (finalImage * 255).astype(np.uint8)
        imageio.imwrite(os.path.join(folderOutput, materialName + "_" + str(idMap) + ".png"), finalImage)

if __name__ == '__main__':
    cropImage("dataExample/woolish.png", "woolish", "cropped")
    runNetwork("cropped", "output_dir", "saved_weights", inputMode="folder")
