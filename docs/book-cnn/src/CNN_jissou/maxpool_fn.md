# Maxpoolレイヤーの実装

## レイヤー構造体の変更

今までのレイヤー構造体は重み(パラメーター)を保持する前提で設計されたものでした。しかし、Maxpoolや今後実装予定の活性化関数単独のレイヤーは重みを保持しません。そこで、保持しない場合に対応できるように変更を加えます。ここでの変更はすべてのレイヤー構造体に対して行います。

### *param_flag* の設定

はじめに重みを持つか持たないかというフラグを構造体内で保持します。このフラグをもとにして処理を分岐させます。また、これを参照する関数をtraitで設定します。

// TODO:コード後で載せる
```rust
impl Layer for Maxpool2d {
    .
    .
    .

    fn has_params(&self) -> bool {
        false   // Maxpoolレイヤーはfalseに、その他のレイヤーはtrueとして初期化
    }
}
```


flagは初期値としてtrue/falseに設定します。Maxpool以外のすべてのレイヤー構造体はtrue、Maxpoolはfalseとして初期化します。

### *paramメソッド* の変更

paramメソッドの戻り値は重みを持つことが前提なので、保持しない場合、エラーが発生します。なので、flagを用いて処理を分岐させます。

// TODO:コード後で載せる
```rust
```


### Optimizerの変更

optimizerの重みを更新する関数も重みを受け取る前提なので、同様にflagで分岐させます。

// TODO:コード後で載せる
```rust
impl Optimizer for SGD {
    fn setup(&mut self, target_model: &impl Model) {
        self.layers = Some(target_model.layers().clone());
    }
    fn update(&mut self) {
        let params = {
            let mut layers = self
                .layers
                .as_ref()
                .ok_or(FrameError::OptimizerError(OptimizerError::MissingModel {
                    optimizer: "SGD",
                }))
                .borrow_mut();

            let mut params_vec: Vec<RcVariable> = Vec::new();

        // filterとhas_parmasを用いて重みをもつレイヤーだけ集める
            for layer in layers.iter_mut().filter(|l| l.has_params()) {
                params_vec.extend(layer.params()?.values().cloned());
            }
            params_vec
        };

        for param in params {
            let new_param = self.update_param(&param);
            param.0.borrow_mut().data = new_param;
        }
    }
}
```

以上でレイヤー構造体の準備は整いました。ではMaxpoolレイヤー構造体のコードを示します。

// TODO:コード後で載せる
```rust
#[derive(Debug, Clone)]
pub struct Maxpool2d {
    input: Option<Weak<RefCell<Variable>>>,
    output: Option<Weak<RefCell<Variable>>>,
    kernel_size: (usize, usize),
    stride_size: (usize, usize),
    pad_size: (usize, usize),
    generation: i32,
    id: usize,
}

impl Layer for Maxpool2d {
    fn set_params(&mut self, _param: &RcVariable) -> FrameResult<()> {
        Err(FrameError::LayerError(LayerError::NoParameterError {
            layer: ("Maxpool2d"),
        })) //Maxpool2dはparamsを持たないので
    }
    fn get_input(&self) -> RcVariable {
        let input = self
            .input
            .as_ref()
            .unwrap()
            .upgrade()
            .as_ref()
            .unwrap()
            .clone();
        RcVariable(input)
    }

    fn get_output(&self) -> RcVariable {
        let output;
        output = self
            .output
            .as_ref()
            .unwrap()
            .upgrade()
            .as_ref()
            .unwrap()
            .clone();

        RcVariable(output)
    }

    fn call(&mut self, input: &RcVariable) -> FrameResult<RcVariable> {
        // inputのvariableからdataを取り出す

        let output = self.forward(input)?;

        //ここから下の処理はbackwardするときだけ必要。

        //　inputを弱参照で覚える
        self.input = Some(input.downgrade());

        //  outputを弱参照(downgrade)で覚える
        self.output = Some(output.downgrade());

        Ok(output)
    }

    fn get_generation(&self) -> i32 {
        self.generation
    }
    fn get_id(&self) -> usize {
        self.id
    }

    fn params(&self) -> FrameResult<&FxHashMap<usize, RcVariable>> {
        Err(FrameError::LayerError(LayerError::NoParameterError {
            layer: ("Maxpool2d"),
        }))
    }
    fn params_mut(&mut self) -> FrameResult<&mut FxHashMap<usize, RcVariable>> {
        Err(FrameError::LayerError(LayerError::NoParameterError {
            layer: ("Maxpool2d"),
        }))
    }

    fn cleargrad(&mut self) {}

    fn has_params(&self) -> bool {
        false
    }
}

impl Maxpool2d {
    fn forward(&mut self, x: &RcVariable) -> RcVariable {
        let y = max_pool2d_simple(x, self.kernel_size, self.stride_size, self.pad_size);

        y
    }

    pub fn new(
        kernel_size: (usize, usize),
        stride_size: (usize, usize),
        pad_size: (usize, usize),
    ) -> Self {
        let maxpool2d = Self {
            input: None,
            output: None,
            kernel_size: kernel_size,
            stride_size: stride_size,
            pad_size: pad_size,
            generation: 0,
            id: id_generator(),
        };

        maxpool2d
    }
}
```

今までのレイヤーとは異なり、重みを持たないので、\\(w\\) の初期化は不要となります。変数の種類が多いのでの、間違いのないよう、引数を設定してください。


ではテストを行います。仮のモデルにこのレイヤーを持たせてデータを流します。先ほどのMaxpool関数のテストと値が等しくなるか確認してください。

// TODO:Tensor変更必要
```rust
#[test]
    fn maxpool2d_layer_test()  {
        use crate::core::TensorToRcVariable;
        use crate::layers as L;
        use crate::models::BaseModel;

        let input_tensor = Tensor::from_vec(
            vec![
                4.0f32, 1.0, 5.0, 3.0, 8.0, 3.0, 2.0, 3.0, 7.0, 2.0, 3.0, 4.0, 1.0, 5.0, 3.0, 9.0,
                4.0, 1.0, 5.0, 3.0, 7.0, 3.0, 2.0, 3.0, 8.0, 2.0, 3.0, 4.0, 1.0, 5.0, 3.0, 9.0,
            ],
            vec![2, 1, 4, 4],
        );

        let kernel_size = (2, 2);
        let stride_size = (2, 2);
        let pad_size = (0, 0);

        let mut model = BaseModel::new();
        model.stack(L::Maxpool2d::new(kernel_size, stride_size, pad_size));

        let input = input_tensor.rv();

        let mut y = model.call(&input);

        println!("y = {}", y.data()); // shape = [1,4,13,13]
        y.backward(false);

        println!("input_grad = {}", input.grad().unwrap().data()); // shape = [1,3,15,15]
    }
```
