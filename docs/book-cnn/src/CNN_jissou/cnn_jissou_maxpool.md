
# Maxpool関数の実装

では続いて **Maxpool関数** を実装していきます。先ほどのConv2d関数と同じ要領で実装してきます。こちらも[Maxpool関数の理論](../CNN_riron/cnn_riron_pool.md) をもとに同じように実装していきます。Maxpoolは重みがないため、変数は少し少なくなります。 

Maxpool関数を実装する前に理論のところで説明した、最大値をとる関数、 **argmax関数** を実装します。この関数に関しては補足のTODO:argmaxで解説していますので、先にこちらで実装、理解しておいてください。

ではここからMaxpool関数を実装していきます。
TODO:コードは後で
```rust
```

計算の流れは理論のところで説明した通りです。argmax関数の引数に注意すれば理論通りの処理をしてくれるはずです。

ではテストを行います。
// TODO:テストコード後で載せる
```rust
#[test]
    fn col2im_function_test() {
        use crate::core_new::ArrayDToRcVariable;

        // im2col_testの出力。(output)
        let input = array![[
            [1.0, 2.0, 3.0, 5.0, 6.0, 7.0, 9.0, 10.0, 11.0],
            [2.0, 3.0, 4.0, 6.0, 7.0, 8.0, 10.0, 11.0, 12.0],
            [5.0, 6.0, 7.0, 9.0, 10.0, 11.0, 13.0, 14.0, 15.0],
            [6.0, 7.0, 8.0, 10.0, 11.0, 12.0, 14.0, 15.0, 16.0]
        ]]
        .rv();

        let kernel_size = (2, 2);
        let stride_size = (1, 1);
        let pad_size = (0, 0);

        let input_shape = [1, 1, 4, 4];

        let mut output = col2im_simple(&input, input_shape, kernel_size, stride_size, pad_size);

        println!("output = {:?}", output);
        /*output = [[[[1.0, 4.0, 6.0, 4.0],
        [10.0, 24.0, 28.0, 16.0],
        [18.0, 40.0, 44.0, 24.0],
        [13.0, 28.0, 30.0, 16.0]]]] */

        output.backward(false);
        println!("input_grad = {:?}", input.grad().unwrap().data());
    }
```


CNNの関数を実装できたので、次はこれらの関数を **レイヤー構造体** として実装していきます。