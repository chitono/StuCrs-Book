# Conv2dレイヤーの実装

ではConv2dをレイヤーとして実装します。先ほどのConv2dを用います。
//TODO:Tensor変更必要
```rust
#[derive(Debug, Clone)]
pub struct Conv2d {
    input: Option<Weak<RefCell<Variable>>>,
    output: Option<Weak<RefCell<Variable>>>,
    out_channels: usize,
    kernel_size: (usize, usize),
    stride_size: (usize, usize),
    pad_size: (usize, usize),
    w_id: Option<usize>,
    b_id: Option<usize>,
    params: FxHashMap<usize, RcVariable>,
    generation: i32,
    id: usize,
}

impl Layer for Conv2d {
    fn set_params(&mut self, param: &RcVariable)  {
        self.params.insert(param.id(), param.clone());
        
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

    fn call(&mut self, input: &RcVariable) -> RcVariable {
        // inputのvariableからdataを取り出す

        let output = self.forward(input)?;

        //ここから下の処理はbackwardするときだけ必要。

        //　inputを弱参照で覚える
        self.input = Some(input.downgrade());

        //  outputを弱参照(downgrade)で覚える
        self.output = Some(output.downgrade());

        output
    }

    fn get_generation(&self) -> i32 {
        self.generation
    }
    fn get_id(&self) -> usize {
        self.id
    }

    fn params(&self) -> &FxHashMap<usize, RcVariable> {
        &self.params
    }
    fn params_mut(&mut self) -> &mut FxHashMap<usize, RcVariable> {
        &mut self.params
    }

    fn cleargrad(&mut self) {
        for (_id, param) in self.params.iter_mut() {
            param.cleargrad();
        }
    }
}

impl Conv2d {
    fn forward(&mut self, x: &RcVariable) -> RcVariable {
        if let None = &self.w_id {
            let c = x.data().shape().dims()[1];
            let oc = self.out_channels;
            let (kh, kw) = self.kernel_size;

            let scale = (1.0f32 / ((c * kh * kw) as f32)).sqrt();

            let w_data = Tensor::standard_normal(vec![oc, c, kh, kw]) * scale.ts();
        
            let w = w_data.rv();

            self.w_id = Some(w.id());
            self.set_params(&w.clone());
        }

        // フィールドでパラメータのidを保持しているので、idでパラメータを呼び出す
        let w_id = self.w_id.unwrap();
        let w = self.params.get(&w_id).unwrap();

        //bはoption型なので、場合分け
        let b;
        if let Some(b_id_data) = self.b_id {
            b = self.params.get(&b_id_data).cloned();
        } else {
            b = None;
        }

        let y = conv2d_simple(x, w, b, self.stride_size, self.pad_size);

        y
    }

    pub fn new(
        out_channels: usize,
        kernel_size: (usize, usize),
        stride_size: (usize, usize),
        pad_size: (usize, usize),
        biased: bool,
    ) -> Self {
        let mut conv2d = Self {
            input: None,
            output: None,
            out_channels: out_channels,
            kernel_size: kernel_size,
            stride_size: stride_size,
            pad_size: pad_size,
            w_id: None,
            b_id: None,
            params: FxHashMap::default(),
            generation: 0,
            id: id_generator(),
        };

        if biased == true {
            let b = Tensor::zeros(vec![1, out_channels as usize, 1, 1]).rv();
            conv2d.b_id = Some(b.id());
            conv2d.set_params(&b.clone());
        }

        conv2d
    }
}

```

レイヤーの実装自体は以前の **Denseレイヤー** などとあまり変わりませんが、重みや引数の設定が少し複雑になります。重みの初期値、次元が全結合の場合と異なります。初期値は正規分布に従うランダムな値であり、次元は4次元になります。

レイヤーで実装してテストを行います。仮のモデルにこのレイヤーを持たせてデータを流します。先ほどのConv2d関数のテストと値が等しくなるか確認してください。

// TODO:Tensor変更必要
```rust
#[test]
fn conv2d_layer_test() {
    use crate::core::TensorToRcVariable;
    use crate::layers as L;
    use crate::models::BaseModel;

    let mut model = BaseModel::new();
    model.stack(L::Conv2d::new(4, (3, 3), (1, 1), (0, 0), false));

    let input_tensor = Tensor::ones(vec![2, 3, 15, 15]);

    let input = input_tensor.rv();

    let mut y = model.call(&input);

    println!("y = {}", y.data()); // shape = [1,4,13,13]
    y.backward(false);

    println!("input_grad = {}", input.grad().unwrap().data()); // shape = [1,3,15,15]
    
}
```
