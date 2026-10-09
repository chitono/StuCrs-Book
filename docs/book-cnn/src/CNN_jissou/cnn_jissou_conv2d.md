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
) -> FrameResult<RcVariable> {
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

    let cols = im2col_simple(input, (kh, kw), stride_size, pad_size)?;

    let weights_2d = weight.reshape(&Shape::new(vec![oc, c * kh * kw])?)?;

    let out = tensordot(&weights_2d, &cols)?;

    let mut out4d = out.reshape(&Shape::new(vec![n, oc, oh, ow])?)?;

    if let Some(b) = bias {
        out4d = out4d + b;
    }

    Ok(out4d)
}
```

変数名や処理の流れは以前の[Conv2d関数の理論](../CNN_riron/cnn_riron_conv.md) をもとにして実装していますので、参照すると理解しやすいです。

Conv2dの処理をかなりシンプルに関数としてまとめることができました。必要な関数をつなげれば簡単にかつ自動的にバックプロパゲーションを行うことができるのが、Function構造体で実装してきた大きなメリットです。またConv2dやMaxpoolといったCNNの関数は引数の種類が多いため、混乱しないように処理の流れを理解しておきましょう。


では計算処理が正しいかテストします。特にバックプロパゲーションがうまく働くか確認します。
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


Conv2d関数を実装できたので、次はもう一つの畳み込みで重要な **Maxpool関数** を実装していきます。