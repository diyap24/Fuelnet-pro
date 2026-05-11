'use strict';

const {
  validateInput,
  preprocessInput,
  denormalizeMpg,
  extendPrediction,
  minMaxNorm,
  minMaxDenorm
} = require('../src/data/preprocessor');

const VALID_INPUT = {
  displacement: 2.5,
  horsepower:   150,
  weight:       3200,
  cylinders:    4,
  modelYear:    2018,
  drivetrain:   1,
  transmission: 0
};

describe('Preprocessor — minMaxNorm', () => {
  test('normalizes min to 0', () => expect(minMaxNorm(0, 0, 100)).toBe(0));
  test('normalizes max to 1', () => expect(minMaxNorm(100, 0, 100)).toBe(1));
  test('normalizes midpoint to 0.5', () => expect(minMaxNorm(50, 0, 100)).toBe(0.5));
  test('handles equal min/max', () => expect(minMaxNorm(5, 5, 5)).toBe(0));
});

describe('Preprocessor — minMaxDenorm', () => {
  test('denormalizes 0 to min', () => expect(minMaxDenorm(0, 10, 60)).toBe(10));
  test('denormalizes 1 to max', () => expect(minMaxDenorm(1, 10, 60)).toBe(60));
  test('roundtrip is idempotent', () => {
    const v = 35, lo = 9, hi = 58;
    expect(minMaxDenorm(minMaxNorm(v, lo, hi), lo, hi)).toBeCloseTo(v, 5);
  });
});

describe('validateInput', () => {
  test('valid input passes', () => {
    const { valid, errors } = validateInput(VALID_INPUT);
    expect(valid).toBe(true);
    expect(errors).toHaveLength(0);
  });

  test('rejects displacement out of range', () => {
    const { valid, errors } = validateInput({ ...VALID_INPUT, displacement: 0.5 });
    expect(valid).toBe(false);
    expect(errors.some(e => e.includes('displacement'))).toBe(true);
  });

  test('rejects horsepower out of range', () => {
    const { valid } = validateInput({ ...VALID_INPUT, horsepower: 600 });
    expect(valid).toBe(false);
  });

  test('rejects invalid drivetrain', () => {
    const { valid, errors } = validateInput({ ...VALID_INPUT, drivetrain: 9 });
    expect(valid).toBe(false);
    expect(errors.some(e => e.includes('drivetrain'))).toBe(true);
  });

  test('rejects future model year', () => {
    const { valid } = validateInput({ ...VALID_INPUT, modelYear: 2030 });
    expect(valid).toBe(false);
  });

  test('collects multiple errors', () => {
    const { errors } = validateInput({ ...VALID_INPUT, displacement: 0.1, horsepower: 9999 });
    expect(errors.length).toBeGreaterThanOrEqual(2);
  });
});

describe('preprocessInput', () => {
  test('returns array of correct length', () => {
    const features = preprocessInput(VALID_INPUT);
    expect(features).toHaveLength(7);
  });

  test('all values in [0, 1] for valid input', () => {
    const features = preprocessInput(VALID_INPUT);
    features.forEach(v => {
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThanOrEqual(1);
    });
  });
});

describe('denormalizeMpg', () => {
  test('denormalizes 0 to min MPG (9)', () => expect(denormalizeMpg(0)).toBe(9));
  test('denormalizes 1 to max MPG (58)', () => expect(denormalizeMpg(1)).toBe(58));
  test('mid norm gives mid MPG', () => expect(denormalizeMpg(0.5)).toBeCloseTo(33.5, 1));
});

describe('extendPrediction', () => {
  test('city mpg is less than combined', () => {
    const { city } = extendPrediction(30);
    expect(city).toBeLessThan(30);
  });
  test('highway mpg is more than combined', () => {
    const { highway } = extendPrediction(30);
    expect(highway).toBeGreaterThan(30);
  });
  test('efficiency class A+ for high MPG', () => {
    expect(extendPrediction(45).efficiencyClass).toBe('A+');
  });
  test('efficiency class F for low MPG', () => {
    expect(extendPrediction(12).efficiencyClass).toBe('F');
  });
  test('co2 inversely proportional to mpg', () => {
    const low  = extendPrediction(15).co2;
    const high = extendPrediction(40).co2;
    expect(low).toBeGreaterThan(high);
  });
  test('annual cost positive number', () => {
    expect(extendPrediction(28).annualFuelCost).toBeGreaterThan(0);
  });
});
