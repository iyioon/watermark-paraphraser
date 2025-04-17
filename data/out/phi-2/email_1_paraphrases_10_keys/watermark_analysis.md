# Watermark Detection Analysis Report

## Summary

- Files analyzed: 10
- Keys tested: [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
- Correct identifications: 7 out of 10
- Accuracy: 70.00%

## Visualization

P-values heatmap for each file and key combination:

![P-values Heatmap](images\heatmap.png)

*Lower p-values (darker colors) indicate stronger evidence of watermarking with that key.*

## Detailed Results

### email-1_key104.txt

- Actual key: 104
- Detected key: 104
- Correct identification: Yes

#### P-values for each key

|   Key |   P-value |
|------:|----------:|
|   100 |  0.633663 |
|   101 |  0.118812 |
|   102 |  0.306931 |
|   103 |  0.891089 |
|   104 |  0.009901 |
|   105 |  0.316832 |
|   106 |  0.990099 |
|   107 |  0.574257 |
|   108 |  0.217822 |
|   109 |  0.752475 |

The key with the lowest p-value is: **104**

✅ Correctly identified the watermark key.

### email-1_key107.txt

- Actual key: 107
- Detected key: 107
- Correct identification: Yes

#### P-values for each key

|   Key |   P-value |
|------:|----------:|
|   100 |  0.188119 |
|   101 |  0.485149 |
|   102 |  0.326733 |
|   103 |  0.376238 |
|   104 |  0.19802  |
|   105 |  0.445545 |
|   106 |  0.277228 |
|   107 |  0.009901 |
|   108 |  0.336634 |
|   109 |  0.366337 |

The key with the lowest p-value is: **107**

✅ Correctly identified the watermark key.

### email-1_key101.txt

- Actual key: 101
- Detected key: 101
- Correct identification: Yes

#### P-values for each key

|   Key |   P-value |
|------:|----------:|
|   100 |  0.39604  |
|   101 |  0.009901 |
|   102 |  0.782178 |
|   103 |  0.306931 |
|   104 |  0.019802 |
|   105 |  0.316832 |
|   106 |  0.089109 |
|   107 |  0.237624 |
|   108 |  0.39604  |
|   109 |  0.821782 |

The key with the lowest p-value is: **101**

✅ Correctly identified the watermark key.

### email-1_key109.txt

- Actual key: 109
- Detected key: 109
- Correct identification: Yes

#### P-values for each key

|   Key |   P-value |
|------:|----------:|
|   100 |  0.584158 |
|   101 |  0.049505 |
|   102 |  0.673267 |
|   103 |  0.811881 |
|   104 |  0.613861 |
|   105 |  0.277228 |
|   106 |  0.90099  |
|   107 |  0.346535 |
|   108 |  0.217822 |
|   109 |  0.009901 |

The key with the lowest p-value is: **109**

✅ Correctly identified the watermark key.

### email-1_key103.txt

- Actual key: 103
- Detected key: 101
- Correct identification: No

#### P-values for each key

|   Key |   P-value |
|------:|----------:|
|   100 |  0.514851 |
|   101 |  0.039604 |
|   102 |  0.435644 |
|   103 |  0.079208 |
|   104 |  0.326733 |
|   105 |  0.138614 |
|   106 |  0.217822 |
|   107 |  0.336634 |
|   108 |  0.207921 |
|   109 |  0.60396  |

The key with the lowest p-value is: **101**

❌ Failed to correctly identify the watermark key.

### email-1_key105.txt

- Actual key: 105
- Detected key: 101
- Correct identification: No

#### P-values for each key

|   Key |   P-value |
|------:|----------:|
|   100 |  0.346535 |
|   101 |  0.049505 |
|   102 |  0.445545 |
|   103 |  0.930693 |
|   104 |  0.217822 |
|   105 |  0.257426 |
|   106 |  0.306931 |
|   107 |  0.247525 |
|   108 |  0.158416 |
|   109 |  0.445545 |

The key with the lowest p-value is: **101**

❌ Failed to correctly identify the watermark key.

### email-1_key102.txt

- Actual key: 102
- Detected key: 102
- Correct identification: Yes

#### P-values for each key

|   Key |   P-value |
|------:|----------:|
|   100 |  0.29703  |
|   101 |  0.118812 |
|   102 |  0.009901 |
|   103 |  0.544554 |
|   104 |  0.435644 |
|   105 |  0.386139 |
|   106 |  0.90099  |
|   107 |  0.455446 |
|   108 |  0.613861 |
|   109 |  0.455446 |

The key with the lowest p-value is: **102**

✅ Correctly identified the watermark key.

### email-1_key108.txt

- Actual key: 108
- Detected key: 108
- Correct identification: Yes

#### P-values for each key

|   Key |   P-value |
|------:|----------:|
|   100 |  0.316832 |
|   101 |  0.247525 |
|   102 |  0.50495  |
|   103 |  0.594059 |
|   104 |  0.534653 |
|   105 |  0.485149 |
|   106 |  0.316832 |
|   107 |  0.524752 |
|   108 |  0.069307 |
|   109 |  0.475248 |

The key with the lowest p-value is: **108**

✅ Correctly identified the watermark key.

### email-1_key106.txt

- Actual key: 106
- Detected key: 101
- Correct identification: No

#### P-values for each key

|   Key |   P-value |
|------:|----------:|
|   100 |  0.19802  |
|   101 |  0.009901 |
|   102 |  0.663366 |
|   103 |  0.613861 |
|   104 |  0.376238 |
|   105 |  0.069307 |
|   106 |  0.049505 |
|   107 |  0.386139 |
|   108 |  0.306931 |
|   109 |  0.455446 |

The key with the lowest p-value is: **101**

❌ Failed to correctly identify the watermark key.

### email-1_key100.txt

- Actual key: 100
- Detected key: 100
- Correct identification: Yes

#### P-values for each key

|   Key |   P-value |
|------:|----------:|
|   100 |  0.019802 |
|   101 |  0.079208 |
|   102 |  0.534653 |
|   103 |  0.782178 |
|   104 |  0.287129 |
|   105 |  0.752475 |
|   106 |  0.643564 |
|   107 |  0.108911 |
|   108 |  0.386139 |
|   109 |  0.762376 |

The key with the lowest p-value is: **100**

✅ Correctly identified the watermark key.

