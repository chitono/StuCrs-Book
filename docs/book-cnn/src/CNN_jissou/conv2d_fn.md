# Conv2dレイヤーの実装

ではConv2dをレイヤーとして実装します。先ほどのConv2dを用います。

// TODO:コード後で載せる
```rust
```


レイヤーの実装自体は以前の **Denseレイヤー** などとあまり変わりませんが、重みや引数の設定が少し複雑になります。重みの初期値、次元が全結合の場合と異なります。初期値は正規分布に従うランダムな値であり、次元は4次元になります。

レイヤーで実装してテストを行います。仮のモデルにこのレイヤーを持たせてデータを流します。先ほどのConv2d関数のテストと値が等しくなるか確認してください。

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
