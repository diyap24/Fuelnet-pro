'use strict';

/**
 * FuelNet Pro — Dataset Seeder
 * Generates a realistic synthetic dataset of 8,428 vehicles
 * and writes it to data/vehicles.csv
 */

const fs = require('fs');
const path = require('path');
const logger = require('../utils/logger');

const OUT_DIR  = path.resolve(__dirname, '../../data');
const OUT_FILE = path.join(OUT_DIR, 'vehicles.csv');

const TOTAL = 8428;

function rand(min, max) {
  return min + Math.random() * (max - min);
}
function randInt(min, max) {
  return Math.floor(rand(min, max + 1));
}
function pick(arr) {
  return arr[Math.floor(Math.random() * arr.length)];
}
function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

/**
 * Physics-inspired MPG formula.
 * Heavier, more powerful, older, larger engines = lower MPG.
 * Newer model years benefit from tech improvements.
 */
function computeMpg(disp, hp, weight, cyl, year, drv, trans) {
  const wNorm = (weight - 1500) / 5000;
  const hNorm = (hp    - 60)   / 440;
  const dNorm = (disp  - 1.0)  / 6.0;
  const yNorm = (year  - 1970) / 54;
  const cNorm = (cyl   - 3)    / 9;

  const drvPenalty   = [0, 0.02, 0.06, 0.10][drv] || 0;
  const transBonus   = [0, 0.04, 0.07][trans]       || 0;

  let mpg = 52
    - 28 * wNorm
    - 18 * hNorm
    - 12 * dNorm
    -  8 * cNorm
    + 10 * yNorm
    +  2 * dNorm * wNorm
    - drvPenalty * 10
    + transBonus * 10;

  // Add realistic noise
  mpg += (Math.random() - 0.5) * 3.5;

  return parseFloat(clamp(mpg, 9, 58).toFixed(1));
}

function generateVehicle() {
  const cylOptions = [3, 4, 4, 4, 4, 6, 6, 8, 8, 10, 12];
  const cyl  = pick(cylOptions);
  const disp = parseFloat(clamp(cyl * 0.4 + rand(-0.3, 0.5), 1.0, 7.0).toFixed(1));
  const hp   = clamp(Math.round(cyl * 35 + rand(-30, 60)), 60, 500);
  const weight = clamp(Math.round(2000 + disp * 300 + rand(-400, 600)), 1500, 6500);
  const year   = randInt(1970, 2024);
  const drv    = randInt(0, 3);
  const trans  = randInt(0, 2);
  const mpg    = computeMpg(disp, hp, weight, cyl, year, drv, trans);

  return { displacement: disp, horsepower: hp, weight, cylinders: cyl, modelYear: year, drivetrain: drv, transmission: trans, mpg };
}

async function seed() {
  if (!fs.existsSync(OUT_DIR)) fs.mkdirSync(OUT_DIR, { recursive: true });

  logger.info(`Seeding dataset: ${TOTAL} records → ${OUT_FILE}`);

  const header = 'displacement,horsepower,weight,cylinders,modelYear,drivetrain,transmission,mpg\n';
  const ws = fs.createWriteStream(OUT_FILE);
  ws.write(header);

  for (let i = 0; i < TOTAL; i++) {
    const v = generateVehicle();
    ws.write(`${v.displacement},${v.horsepower},${v.weight},${v.cylinders},${v.modelYear},${v.drivetrain},${v.transmission},${v.mpg}\n`);
  }

  ws.end();
  await new Promise(res => ws.on('finish', res));
  logger.info('Dataset seeding complete.');
}

seed().catch(err => { logger.error(err); process.exit(1); });
