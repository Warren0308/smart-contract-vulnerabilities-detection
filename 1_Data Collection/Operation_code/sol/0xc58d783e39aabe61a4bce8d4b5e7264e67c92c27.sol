PUSH1 0x60<br>PUSH1 0x40<br>MSTORE<br>PUSH1 0xe0<br>PUSH1 0x02<br>EXP<br>PUSH1 0x00<br>CALLDATALOAD<br>DIV<br>PUSH4 0x2e6e504a<br>DUP2<br>EQ<br>PUSH2 0x003c<br>JUMPI<br>DUP1<br>PUSH4 0x3ccfd60b<br>EQ<br>PUSH2 0x0131<br>JUMPI<br>DUP1<br>PUSH4 0xeedcf50a<br>EQ<br>PUSH2 0x0234<br>JUMPI<br>DUP1<br>PUSH4 0xfdf97cb2<br>EQ<br>PUSH2 0x024f<br>JUMPI<br>JUMPDEST<br>STOP<br>JUMPDEST<br>PUSH2 0x003a<br>PUSH32 0x18160ddd00000000000000000000000000000000000000000000000000000000<br>PUSH1 0x60<br>SWAP1<br>DUP2<br>MSTORE<br>PUSH20 0xda4a4626d3e16e094de3225a751aab7128e96526<br>SWAP1<br>PUSH1 0x00<br>SWAP1<br>PUSH20 0x35a051a0010aba705c9008d7a7eff6fb88f6ea7b<br>SWAP1<br>PUSH4 0x18160ddd<br>SWAP1<br>PUSH1 0x64<br>SWAP1<br>PUSH1 0x20<br>SWAP1<br>PUSH1 0x04<br>DUP2<br>DUP8<br>DUP8<br>PUSH2 0x61da<br>GAS<br>SUB<br>CALL<br>ISZERO<br>PUSH2 0x0002<br>JUMPI<br>POP<br>POP<br>PUSH1 0x40<br>DUP1<br>MLOAD<br>DUP1<br>MLOAD<br>PUSH1 0xe0<br>PUSH1 0x02<br>EXP<br>PUSH4 0x70a08231<br>MUL<br>DUP3<br>MSTORE<br>ADDRESS<br>PUSH1 0x01<br>PUSH1 0xa0<br>PUSH1 0x02<br>EXP<br>SUB<br>AND<br>PUSH1 0x04<br>DUP4<br>ADD<br>MSTORE<br>SWAP2<br>MLOAD<br>SWAP2<br>SWAP3<br>SWAP2<br>PUSH4 0x70a08231<br>SWAP2<br>PUSH1 0x24<br>DUP2<br>DUP2<br>ADD<br>SWAP3<br>PUSH1 0x20<br>SWAP3<br>SWAP1<br>SWAP2<br>SWAP1<br>DUP3<br>SWAP1<br>SUB<br>ADD<br>DUP2<br>DUP9<br>DUP8<br>PUSH2 0x61da<br>GAS<br>SUB<br>CALL<br>ISZERO<br>PUSH2 0x0002<br>JUMPI<br>POP<br>POP<br>POP<br>PUSH1 0x40<br>MLOAD<br>DUP1<br>MLOAD<br>SWAP1<br>PUSH1 0x20<br>ADD<br>POP<br>ADDRESS<br>PUSH1 0x01<br>PUSH1 0xa0<br>PUSH1 0x02<br>EXP<br>SUB<br>AND<br>BALANCE<br>ADD<br>SUB<br>PUSH1 0x40<br>MLOAD<br>DUP1<br>SWAP1<br>POP<br>PUSH1 0x00<br>PUSH1 0x40<br>MLOAD<br>DUP1<br>DUP4<br>SUB<br>DUP2<br>DUP6<br>DUP9<br>DUP9<br>CALL<br>POP<br>POP<br>POP<br>POP<br>POP<br>JUMP<br>JUMPDEST<br>PUSH2 0x003a<br>PUSH1 0xe0<br>PUSH1 0x02<br>EXP<br>PUSH4 0x70a08231<br>MUL<br>PUSH1 0x60<br>SWAP1<br>DUP2<br>MSTORE<br>CALLER<br>PUSH1 0x01<br>PUSH1 0xa0<br>PUSH1 0x02<br>EXP<br>SUB<br>AND<br>PUSH1 0x64<br>MSTORE<br>PUSH1 0x00<br>SWAP1<br>PUSH20 0x35a051a0010aba705c9008d7a7eff6fb88f6ea7b<br>SWAP1<br>PUSH4 0x70a08231<br>SWAP1<br>PUSH1 0x84<br>SWAP1<br>PUSH1 0x20<br>SWAP1<br>PUSH1 0x24<br>DUP2<br>DUP8<br>DUP8<br>PUSH2 0x61da<br>GAS<br>SUB<br>CALL<br>ISZERO<br>PUSH2 0x0002<br>JUMPI<br>POP<br>POP<br>PUSH1 0x40<br>DUP1<br>MLOAD<br>DUP1<br>MLOAD<br>PUSH32 0x23b872dd00000000000000000000000000000000000000000000000000000000<br>DUP3<br>MSTORE<br>CALLER<br>PUSH1 0x01<br>PUSH1 0xa0<br>PUSH1 0x02<br>EXP<br>SUB<br>SWAP1<br>DUP2<br>AND<br>PUSH1 0x04<br>DUP5<br>ADD<br>MSTORE<br>ADDRESS<br>AND<br>PUSH1 0x24<br>DUP4<br>ADD<br>MSTORE<br>PUSH1 0x44<br>DUP3<br>ADD<br>DUP2<br>SWAP1<br>MSTORE<br>SWAP2<br>MLOAD<br>SWAP2<br>SWAP4<br>PUSH4 0x23b872dd<br>SWAP3<br>PUSH1 0x64<br>DUP4<br>DUP2<br>ADD<br>SWAP4<br>PUSH1 0x20<br>SWAP4<br>SWAP1<br>DUP4<br>SWAP1<br>SUB<br>SWAP1<br>SWAP2<br>ADD<br>SWAP1<br>DUP3<br>SWAP1<br>DUP8<br>PUSH2 0x61da<br>GAS<br>SUB<br>CALL<br>ISZERO<br>PUSH2 0x0002<br>JUMPI<br>POP<br>POP<br>PUSH1 0x40<br>MLOAD<br>MLOAD<br>ISZERO<br>SWAP1<br>POP<br>DUP1<br>PUSH2 0x022a<br>JUMPI<br>POP<br>PUSH1 0x40<br>MLOAD<br>CALLER<br>PUSH1 0x01<br>PUSH1 0xa0<br>PUSH1 0x02<br>EXP<br>SUB<br>AND<br>SWAP1<br>PUSH1 0x00<br>SWAP1<br>DUP4<br>SWAP1<br>DUP3<br>DUP2<br>DUP2<br>DUP2<br>DUP6<br>DUP9<br>DUP4<br>CALL<br>SWAP4<br>POP<br>POP<br>POP<br>POP<br>ISZERO<br>JUMPDEST<br>ISZERO<br>PUSH2 0x027d<br>JUMPI<br>PUSH2 0x0002<br>JUMP<br>JUMPDEST<br>PUSH2 0x026a<br>PUSH20 0x35a051a0010aba705c9008d7a7eff6fb88f6ea7b<br>DUP2<br>JUMP<br>JUMPDEST<br>PUSH2 0x026a<br>PUSH20 0xda4a4626d3e16e094de3225a751aab7128e96526<br>DUP2<br>JUMP<br>JUMPDEST<br>PUSH1 0x01<br>PUSH1 0xa0<br>PUSH1 0x02<br>EXP<br>SUB<br>AND<br>PUSH1 0x60<br>SWAP1<br>DUP2<br>MSTORE<br>PUSH1 0x20<br>SWAP1<br>RETURN<br>JUMPDEST<br>POP<br>JUMP<br>