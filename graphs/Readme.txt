explanation of the graphs:

the accuracy is the loss ~approximitely 2.x million

32000 graph increases at increments of 100
9000 increases at increments of 10
2500 = increments of 3
*did not go by the mutiple 2's , but img3_super_zoomed is interesting


img3:
imput size changes from 16 -> x = {32000, 9000, 2500}
feature size = 1
decoder_int = 8
seed = 116 (the best seed i found for accuracy; tested seeds 1 - 200)

feature4:
imput size changes from 16 -> x = {32000, 9000, 2500}
feature size = 4
decoder_int = 8
seed = 116 (the best seed i found for accuracy; tested seeds 1 - 200)


##########################################33
information for the epoch graphs.
The graphs are defined by the datasets i have below, based on input size, feature size, and decoder_int1.

Epoch_error_graph.jpg
| *        | input_size | feature_size | decoder_int1 |
| -------- | ---------- | ------------ | ------------ |
| dataset1 |     8      |      8       |       4      |
| dataset2 |     16     |      8       |       8      |
| dataset3 |     16     |      16      |       8      |
| dataset4 |     32     |      1       |       8      |
| dataset5 |     32     |      2       |       8      |
| dataset6 |     32     |      2       |       16     |
| dataset7 |     32     |      16      |       16     |
| dataset8 |     32     |      21      |       21     |
| dataset9 |     8      |      1       |       8      |




Epoch_error_graph_2.jpg
| *        | input_size | feature_size | decoder_int1 |
| -------- | ---------- | ------------ | ------------ |
| dataset1 |     8      |      8       |       4      |
| dataset2 |     8      |      8       |       8      |
| dataset3 |     8      |      1       |       4      |
| dataset4 |     8      |      4       |       4      |
| dataset5 |     8      |      8       |       8      |
| dataset6 |     8      |      1       |       6      |
| dataset7 |     8      |      4       |       6      |
| dataset8 |     4      |      4       |       8      |
| dataset9 |     2      |      2       |       8      |

Epoch_error_graph_3.jpg
| *        | input_size | feature_size | decoder_int1 |
| -------- | ---------- | ------------ | ------------ |
| dataset1 |     8      |      8       |       4      |
| dataset2 |     64     |      32      |       32     |
| dataset3 |     64     |      16      |       32     |
| dataset4 |     64     |      64      |       16     |
| dataset5 |     64     |      8       |       8      |
| dataset6 |     128    |      64      |       64     |
| dataset7 |     128    |      16      |       16     |
| dataset8 |     256    |      128     |       128    |
| dataset9 |     256    |      64      |       16     |

Epoch_error_graph_4.jpg
| *        | input_size | feature_size | decoder_int1 |
| -------- | ---------- | ------------ | ------------ |
| dataset1 |     8      |      8       |       4      |
| dataset2 |     64     |      12      |       8      |
| dataset3 |     64     |      20      |       4      |
| dataset4 |     64     |      16      |       4      |
| dataset5 |     64     |      32      |       16     |
| dataset6 |     64     |      8       |       12     |
| dataset7 |     64     |      32      |       64     |
| dataset8 |     64     |      8       |       12     |
| dataset9 |     64     |      4       |       16     |

Epoch_error_graph_5.jpg
| *        | input_size | feature_size | decoder_int1 |
| -------- | ---------- | ------------ | ------------ |
| dataset1 |     8      |      8       |       4      |
| dataset2 |     16     |      8       |       4      |
| dataset3 |     16     |      8       |       8      |
| dataset4 |     16     |      4       |       8      |
| dataset5 |     16     |      4       |       4      |
| dataset6 |     16     |      10      |       5      |
| dataset7 |     16     |      1       |       4      |
| dataset8 |     16     |      1       |       8      |
| dataset9 |     16     |      8       |       16     |

