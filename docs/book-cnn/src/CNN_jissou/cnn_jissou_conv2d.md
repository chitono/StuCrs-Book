# Conv2d関数の実装

では  **Conv2d関数** を実装していきます。ここでは前に説明した呼び出す関数として説明します。  



Conv2dを呼び出す関数 *Conv2d_simple* を実装してみましょう。Conv2dを構成する関数はすべてFunction構造体で実装してきたので、それらをつなげて一つの関数として実装できます。

```rust
pub fn conv2d_simple(
    input: &RcVariable,
    weight: &RcVariable,
    bias: Option<RcVariable>,
    stride_size: (usize, usize),
    pad_size: (usize, usize),
) -> RcVariable {
    let input_data = input.data();
    let weight_data = weight.data();

    let input_shape = input_data.shape().dims();
    let weight_shape = weight_data.shape().dims();

    let n = input_shape[0];
    let c = input_shape[1];
    let h = input_shape[2];
    let w = input_shape[3];

    // weightから形状のデータを取り出す。
    let oc = weight_shape[0];
    let c_wt = weight_shape[1];
    let kh = weight_shape[2];
    let kw = weight_shape[3];

    // チャンネル数がinputとweightで一致しているか確認。
    if c != c_wt {
        panic!("Conv2d: inputのチャンネル数とweightのチャンネル数が一致しません。");
    }

    let (oh, ow) = get_conv_outsize((h, w), (kh, kw), stride_size, pad_size);

    let cols = im2col_simple(input, (kh, kw), stride_size, pad_size);

    let weights_2d = weight.reshape(&Shape::new(vec![oc, c * kh * kw]));

    let out = tensordot(&weights_2d, &cols);

    let mut out4d = out.reshape(&Shape::new(vec![n, oc, oh, ow]));

    if let Some(b) = bias {
        out4d = out4d + b;
    }

    out4d
}
```

変数名や処理の流れは以前の[Conv2d関数の理論](../CNN_riron/cnn_riron_conv.md) をもとにして実装していますので、参照すると理解しやすいです。

Conv2dの処理をかなりシンプルに関数としてまとめることができました。必要な関数をつなげれば簡単にかつ自動的にバックプロパゲーションを行うことができるのが、Function構造体で実装してきた大きなメリットです。またConv2dやMaxpoolといったCNNの関数は引数の種類が多いため、混乱しないように処理の流れを理解しておきましょう。


では計算処理が正しいかテストします。特にバックプロパゲーションがうまく働くか確認します。
```rust
#[test]
    fn conv2d_test()  {
        use crate::core::TensorToRcVariable;

        let input_tensor = Tensor::ones(vec![2, 5, 15, 15]);
        let weight_tensor = Tensor::ones(vec![8, 5, 3, 3]);

        let input = input_tensor.rv();
        let weight = weight_tensor.rv();

        let stride_size = (1, 1);
        let pad_size = (0, 0);

        let mut output = conv2d_simple(&input, &weight, None, stride_size, pad_size);

        println!("output_shape = {:?}", output.data().shape()); //shape = (1,8,15,15)

        output.backward(false);

        println!(
            "input_grad_shape = {:?}",
            input.grad().unwrap().data().to_vec()
        ); //shape = (1,5,15,15)

        
    }
```


Conv2d関数を実装できたので、次はもう一つの畳み込みで重要な **Maxpool関数** を実装していきます。