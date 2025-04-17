# Watermark Detection Analysis Report

## Summary

- Files analyzed: 3
- Keys tested: [1, 2, 3]
- Correct identifications: 3 out of 3
- Accuracy: 100.00%

## Visualization

P-values heatmap for each file and key combination:

![P-values Heatmap](images\heatmap.png)

*Lower p-values (darker colors) indicate stronger evidence of watermarking with that key.*

## Detailed Results

### email-1_key3.txt

- Actual key: 3
- Detected key: 3
- Correct identification: Yes

#### P-values for each key

|   Key |   P-value |
|------:|----------:|
|     1 |  0.742574 |
|     2 |  0.415842 |
|     3 |  0.009901 |

The key with the lowest p-value is: **3**

✅ Correctly identified the watermark key.

### email-1_key2.txt

- Actual key: 2
- Detected key: 2
- Correct identification: Yes

#### P-values for each key

|   Key |   P-value |
|------:|----------:|
|     1 |  0.693069 |
|     2 |  0.019802 |
|     3 |  0.19802  |

The key with the lowest p-value is: **2**

✅ Correctly identified the watermark key.

### email-1_key1.txt

- Actual key: 1
- Detected key: 1
- Correct identification: Yes

#### P-values for each key

|   Key |   P-value |
|------:|----------:|
|     1 |  0.138614 |
|     2 |  0.584158 |
|     3 |  0.316832 |

The key with the lowest p-value is: **1**

✅ Correctly identified the watermark key.

