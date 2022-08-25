#ifndef NPKIT_EVENT_H_
#define NPKIT_EVENT_H_

#define NPKIT_EVENT_INVALID                                     0x0

#define NPKIT_EVENT_SEND_ENTRY                                  0x1
#define NPKIT_EVENT_SEND_EXIT                                   0x2
#define NPKIT_EVENT_SEND_FROM_OUTPUT_ENTRY                      0x3
#define NPKIT_EVENT_SEND_FROM_OUTPUT_EXIT                       0x4
#define NPKIT_EVENT_DIRECT_SEND_ENTRY                           0x5
#define NPKIT_EVENT_DIRECT_SEND_EXIT                            0x6
#define NPKIT_EVENT_DIRECT_SEND_FROM_OUTPUT_ENTRY               0x7
#define NPKIT_EVENT_DIRECT_SEND_FROM_OUTPUT_EXIT                0x8

#define NPKIT_EVENT_RECV_ENTRY                                  0x9
#define NPKIT_EVENT_RECV_EXIT                                   0xA
#define NPKIT_EVENT_DIRECT_RECV_ENTRY                           0xB
#define NPKIT_EVENT_DIRECT_RECV_EXIT                            0xC

#define NPKIT_EVENT_REDUCE_ENTRY                                0xD
#define NPKIT_EVENT_REDUCE_EXIT                                 0xE

#define NPKIT_EVENT_LOCAL_COPY_ENTRY                            0xF
#define NPKIT_EVENT_LOCAL_COPY_EXIT                             0x10

#define NPKIT_EVENT_COPY_SEND_ENTRY                             0x11
#define NPKIT_EVENT_COPY_SEND_EXIT                              0x12
#define NPKIT_EVENT_DIRECT_COPY_SEND_ENTRY                      0x13
#define NPKIT_EVENT_DIRECT_COPY_SEND_EXIT                       0x14

#define NPKIT_EVENT_RECV_COPY_SEND_ENTRY                        0x15
#define NPKIT_EVENT_RECV_COPY_SEND_EXIT                         0x16
#define NPKIT_EVENT_DIRECT_RECV_COPY_SEND_ENTRY                 0x17
#define NPKIT_EVENT_DIRECT_RECV_COPY_SEND_EXIT                  0x18
#define NPKIT_EVENT_RECV_COPY_DIRECT_SEND_ENTRY                 0x19
#define NPKIT_EVENT_RECV_COPY_DIRECT_SEND_EXIT                  0x1A

#define NPKIT_EVENT_RECV_REDUCE_COPY_ENTRY                      0x1B
#define NPKIT_EVENT_RECV_REDUCE_COPY_EXIT                       0x1C

#define NPKIT_EVENT_RECV_REDUCE_SEND_ENTRY                      0x1D
#define NPKIT_EVENT_RECV_REDUCE_SEND_EXIT                       0x1E
#define NPKIT_EVENT_DIRECT_RECV_REDUCE_SEND_ENTRY               0x1F
#define NPKIT_EVENT_DIRECT_RECV_REDUCE_SEND_EXIT                0x20

#define NPKIT_EVENT_RECV_REDUCE_COPY_SEND_ENTRY                 0x21
#define NPKIT_EVENT_RECV_REDUCE_COPY_SEND_EXIT                  0x22
#define NPKIT_EVENT_DIRECT_RECV_REDUCE_COPY_SEND_ENTRY          0x23
#define NPKIT_EVENT_DIRECT_RECV_REDUCE_COPY_SEND_EXIT           0x24

#define NPKIT_EVENT_NET_SEND_ENTRY                              0x25
#define NPKIT_EVENT_NET_SEND_EXIT                               0x26
#define NPKIT_EVENT_NET_RECV_ENTRY                              0x27
#define NPKIT_EVENT_NET_RECV_EXIT                               0x28

#define NPKIT_EVENT_DEP_CHECK_ENTRY                             0x29
#define NPKIT_EVENT_DEP_CHECK_EXIT                              0x2A

#define NPKIT_EVENT_TIME_SYNC_GPU                               0x2B
#define NPKIT_EVENT_TIME_SYNC_CPU                               0x2C

#endif
