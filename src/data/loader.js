'use strict';

const fs   = require('fs');
const path = require('path');
const csv  = require('csv-parser');
const tf = require('@tensorflow/tfjs');const { preprocessRow } = require('./preprocessor');
const logger = require('../utils/logger');

const DEFAULT_CSV = path.resolve(__dirname, '../../data/vehicles.csv');

/**
 * Load and parse the CSV dataset into raw JS arrays.
 */
function loadCSV(filePath = DEFAULT_CSV) {
  return new Promise((resolve, reject) => {
    if (!fs.existsSync(filePath)) {
      return reject(new Error(`Dataset not found at ${filePath}. Run: npm run seed`));
    }
    const rows = [];
    fs.createReadStream(filePath)
      .pipe(csv())
      .on('data', row => rows.push(row))
      .on('end',  () => resolve(rows))
      .on('error', reject);
  });
}

/**
 * Shuffle array in-place (Fisher-Yates).
 */
function shuffle(arr) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

/**
 * Load dataset and return train/test split as TensorFlow tensors.
 * @param {number} testSplit  Fraction for test set (default 0.2)
 * @param {string} csvPath    Path to CSV file
 */
async function loadDataset(testSplit = 0.2, csvPath = DEFAULT_CSV) {
  logger.info('Loading dataset...', { path: csvPath });
  const raw = await loadCSV(csvPath);
  logger.info(`Loaded ${raw.length} records`);

  shuffle(raw);

  const processed = raw.map(preprocessRow);

  const splitIdx  = Math.floor(processed.length * (1 - testSplit));
  const trainData = processed.slice(0, splitIdx);
  const testData  = processed.slice(splitIdx);

  logger.info(`Split: ${trainData.length} train / ${testData.length} test`);

  const toTensors = (data) => {
    const xs = tf.tensor2d(data.map(d => d.features));
    const ys = tf.tensor2d(data.map(d => [d.label]));
    return { xs, ys };
  };

  return {
    train: toTensors(trainData),
    test:  toTensors(testData),
    total: raw.length
  };
}

module.exports = { loadDataset, loadCSV, shuffle };
