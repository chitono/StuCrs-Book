
# Maxpool関数の実装

では続いて **Maxpool関数** を実装していきます。先ほどのConv2d関数と同じ要領で実装してきます。こちらも[Maxpool関数の理論](../CNN_riron/cnn_riron_pool.md) をもとに同じように実装していきます。Maxpoolは重みがないため、変数は少し少なくなります。 

Maxpool関数を実装する前に理論のところで説明した、最大値をとる関数、 **argmax関数** を実装します。この関数に関しては補足のTODO:argmaxで解説していますので、先にこちらで実装、理解しておいてください。

ではここからMaxpool関数を実装していきます。
```rust
pub fn max_pool2d_simple(
    input: &RcVariable,
    kernel_size: (usize, usize),
    stride_size: (usize, usize),
    pad_size: (usize, usize),
) {
    let input_data = input.data();

    let input_shape = input_data.shape().dims();

    let n = input_shape[0];
    let c = input_shape[1];
    let h = input_shape[2];
    let w = input_shape[3];

    let (kh, kw) = kernel_size;

    let (oh, ow) = get_conv_outsize((h, w), kernel_size, stride_size, pad_size);

    let cols = im2col_simple(input, kernel_size, stride_size, pad_size);

    let cols = cols.reshape(&Shape::new(vec![n, kh * kw, c * oh * ow]));

    let y = max(&cols, Some(1));

    let output = y
        .reshape(&Shape::new(vec![n, oh, ow, c]))
        .permute_axes(vec![0, 3, 1, 2]);

    output
}
```

計算の流れは理論のところで説明した通りです。argmax関数の引数に注意すれば理論通りの処理をしてくれるはずです。

ではテストを行います。
```rust
#[test]
    fn max_pool2d_test() {
        use crate::core::TensorToRcVariable;

        let input_tensor = Tensor::from_vec(
            vec![
                4.0f32, 1.0, 5.0, 3.0, 7.0, 3.0, 2.0, 3.0, 7.0, 2.0, 3.0, 4.0, 1.0, 5.0, 3.0, 9.0,
                4.0, 1.0, 5.0, 3.0, 7.0, 3.0, 2.0, 3.0, 7.0, 2.0, 3.0, 4.0, 1.0, 5.0, 3.0, 9.0,
            ],
            vec![2, 1, 4, 4],
        );

        println!("input_shape = {:?}", input_tensor.shape());

        let input = input_tensor.rv();
        let kernel_size = (2, 2);
        let stride_size = (2, 2);
        let pad_size = (0, 0);

        let mut output = max_pool2d_simple(&input, kernel_size, stride_size, pad_size);

        println!("output = {}", output.data()); //shape = (1,2,3,3)

        output.backward(false);

        println!("input_grad= {}", input.grad().unwrap().data()); //shape = (1,5,15,15)

    }
```


CNNの関数を実装できたので、次はこれらの関数を **レイヤー構造体** として実装していきます。