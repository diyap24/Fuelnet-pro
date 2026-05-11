'use strict';

/**
 * Integration tests for the FuelNet API routes.
 * These tests mock the prediction service to avoid needing a trained model.
 */

const request = require('supertest');

jest.mock('../components/predictionService', () => ({
  ready: true,
  initialize: jest.fn().mockResolvedValue(undefined),
  predictOne: jest.fn().mockResolvedValue({
    predicted_mpg:   28.4,
    city_mpg:        24.1,
    highway_mpg:     33.2,
    co2_g_per_mile:  313,
    efficiency_class: 'B',
    annual_fuel_cost_usd: 1875,
    feature_importance: {},
    model_info: { architecture: '7→64→128→64→1', r2_score: 0.936 },
    latency_ms: 2
  }),
  predictBatch: jest.fn().mockResolvedValue([])
}));

const app = require('../server');

describe('GET /api/health', () => {
  it('returns 200 with status ok', async () => {
    const res = await request(app).get('/api/health');
    expect(res.statusCode).toBe(200);
    expect(res.body.status).toBe('ok');
  });
});

describe('GET /api/model/info', () => {
  it('returns model metadata', async () => {
    const res = await request(app).get('/api/model/info');
    expect(res.statusCode).toBe(200);
    expect(res.body.name).toBe('FuelNet Pro');
    expect(res.body.input_features).toHaveLength(7);
  });
});

describe('POST /api/predict', () => {
  const validPayload = {
    displacement: 2.5, horsepower: 150, weight: 3200,
    cylinders: 4, modelYear: 2018, drivetrain: 1, transmission: 0
  };

  it('returns 200 with valid payload', async () => {
    const res = await request(app).post('/api/predict').send(validPayload);
    expect(res.statusCode).toBe(200);
    expect(res.body.success).toBe(true);
    expect(res.body.data.predicted_mpg).toBeDefined();
  });

  it('returns efficiency class', async () => {
    const res = await request(app).post('/api/predict').send(validPayload);
    expect(res.body.data.efficiency_class).toBeDefined();
  });
});

describe('POST /api/predict/batch', () => {
  it('rejects empty vehicles array', async () => {
    const res = await request(app).post('/api/predict/batch').send({ vehicles: [] });
    expect(res.statusCode).toBe(400);
  });

  it('rejects batch over 100', async () => {
    const vehicles = Array(101).fill({ displacement: 2.5, horsepower: 150, weight: 3200, cylinders: 4, modelYear: 2018, drivetrain: 1, transmission: 0 });
    const res = await request(app).post('/api/predict/batch').send({ vehicles });
    expect(res.statusCode).toBe(400);
  });
});
