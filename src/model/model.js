'use strict';
const tf   = require('@tensorflow/tfjs');
const path = require('path');
const fs   = require('fs');
const logger = require('../utils/logger');

const MODEL_DIR = path.resolve(process.env.MODEL_SAVE_PATH || './models/fuelnet_model');

function buildModel({ units = [64, 128, 64], learningRate = 0.001, dropoutRate = 0.2, inputSize = 7 } = {}) {
  const model = tf.sequential({ name: 'FuelNet_v2' });

  model.add(tf.layers.dense({
    inputShape: [inputSize], units: units[0], activation: 'relu',
    kernelRegularizer: tf.regularizers.l2({ l2: 1e-4 }), name: 'hidden_1'
  }));
  model.add(tf.layers.dropout({ rate: dropoutRate, name: 'drop_1' }));

  for (let i = 1; i < units.length; i++) {
    model.add(tf.layers.dense({
      units: units[i], activation: 'relu',
      kernelRegularizer: tf.regularizers.l2({ l2: 1e-4 }), name: `hidden_${i + 1}`
    }));
    model.add(tf.layers.dropout({ rate: dropoutRate, name: `drop_${i + 1}` }));
  }

  model.add(tf.layers.dense({ units: 1, activation: 'linear', name: 'output' }));

  model.compile({
    optimizer: tf.train.adam(learningRate),
    loss: 'meanSquaredError',
    metrics: ['mae']
  });

  logger.info('Model built', { params: model.countParams() });
  return model;
}

async function saveModel(model, dir = MODEL_DIR) {
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });

  // Save weights manually as JSON (works without tfjs-node)
  const weightData = [];
  for (const layer of model.layers) {
    const weights = layer.getWeights();
    if (weights.length > 0) {
      weightData.push({
        name: layer.name,
        weights: weights.map(w => ({
          shape: w.shape,
          data: Array.from(w.dataSync())
        }))
      });
    }
  }

  const saveObj = {
    modelConfig: JSON.parse(model.toJSON()),
    weightData,
    savedAt: new Date().toISOString()
  };

  fs.writeFileSync(path.join(dir, 'model_weights.json'), JSON.stringify(saveObj));
  logger.info('Model saved', { path: dir });
}

async function loadModel(dir = MODEL_DIR) {
  const weightsPath = path.join(dir, 'model_weights.json');
  if (!fs.existsSync(weightsPath)) {
    logger.warn('No saved model found. Run: npm run train');
    return null;
  }

  const saveObj = JSON.parse(fs.readFileSync(weightsPath, 'utf8'));

  // Rebuild model from config
  const model = await tf.models.modelFromJSON(saveObj.modelConfig);

  // Restore weights
  for (const layerData of saveObj.weightData) {
    const layer = model.getLayer(layerData.name);
    if (layer) {
      const tensors = layerData.weights.map(w =>
        tf.tensor(w.data, w.shape)
      );
      layer.setWeights(tensors);
      tensors.forEach(t => t.dispose());
    }
  }

  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: 'meanSquaredError',
    metrics: ['mae']
  });

  logger.info('Model loaded from disk');
  return model;
}

function predict(model, normalizedFeatures) {
  return tf.tidy(() => {
    const input  = tf.tensor2d([normalizedFeatures]);
    const output = model.predict(input);
    return output.dataSync()[0];
  });
}

module.exports = { buildModel, saveModel, loadModel, predict };