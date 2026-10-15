import {root, zto_random, round_float, absolute_value, closest_pow_of_two} from '../src/math.js'

test('root', () => {
    expect(root(4, 2)).toBe(2);
});

test('round_float', () => {
    expect(round_float(.5)).toBe(1);
});

test('absolute_value', () => {
    expect(absolute_value(-3)).toBe(3);
});

test('closest_pow_of_two', () => {
    expect(closest_pow_of_two(234.123)).toBe(256);
});

