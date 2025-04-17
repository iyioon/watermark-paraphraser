# Watermark Detection Analysis Report

## Summary

- Files analyzed: 5
- Keys tested: [10, 11, 12, 13, 14]
- Correct identifications: 3 out of 5
- Accuracy: 60.00%

## Visualization

P-values heatmap for each file and key combination:

![P-values Heatmap](images\heatmap.png)

*Lower p-values (darker colors) indicate stronger evidence of watermarking with that key.*

## Detailed Results

### email-1_key11.txt

- Actual key: 11
- Detected key: 11
- Correct identification: Yes

#### P-values for each key

|   Key |   P-value |
|------:|----------:|
|    10 |  0.891089 |
|    11 |  0.009901 |
|    12 |  0.90099  |
|    13 |  0.752475 |
|    14 |  0.019802 |

The key with the lowest p-value is: **11**

