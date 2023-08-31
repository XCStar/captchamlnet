#r "nuget: Microsoft.ML.OnnxTransformer, 2.0.1"
#r "nuget: SixLabors.ImageSharp, 3.0.2"
#r "nuget: Microsoft.ML.OnnxRuntime, 1.15.1"
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.Tokenizers;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.ColorSpaces;
using SixLabors.ImageSharp.Formats;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using System.Linq;
var testPath="";
var modelPath ="";

var files = Directory.GetFiles(testPath, "*.jpg");
foreach (var item in files)
{
    var dist=Path.GetFileNameWithoutExtension(item).Split("_")[0];
    var result=Test(item);
     Console.WriteLine($"预测值:{dist} 预测结果：{result}  :{result==dist}");

}


string Test(string path)
{
    using Image<Rgb24> image = Image.Load<Rgb24>(path);
    //image.Metadata.DecodedImageFormat.Dump();
    var format = image.Metadata.DecodedImageFormat;
    using var stream = new MemoryStream();
    image.Mutate(x =>
    {
        x.Resize(new ResizeOptions
        {
            Size = new Size(160, 60),
            Mode = ResizeMode.Crop
        });

    });
    image.Save(stream, format);
    var mean = new[] { 0.485f, 0.485f, 0.406f };
    var stddev = new[] { 0.229f, 0.224f, 0.225f };
    Tensor<float> input = new DenseTensor<float>(new int[] { 1, 3, 60, 160 });
    //ReadFile(input);
    //归一化处理方式一定要和训练模型一致,否则无法识别
    image.ProcessPixelRows(accssor =>
    {
        //Console.WriteLine($"width:{accssor.Width} height:{accssor.Height} {input.Length}");
        for (int i = 0; i < accssor.Height; i++)
        {
            var pixelSpan = accssor.GetRowSpan(i);
            for (int j = 0; j < accssor.Width; j++)
            {
              
                input[0, 0, i, j] = pixelSpan[j].R / 255f;//((pixelSpan[i].R / 255f) - mean[0]) / stddev[0];
                input[0, 1, i, j] = pixelSpan[j].G / 255f;//((pixelSpan[i].G / 255f) - mean[1]) / stddev[1];
                input[0, 2, i, j] = pixelSpan[j].B / 255f;//((pixelSpan[i].B / 255f) - mean[2]) / stddev[2];
                
            }
        }

    });
    using var session = new InferenceSession(modelPath);
    var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(session.InputNames.FirstOrDefault(), input) };


    using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = session.Run(inputs);
    //results.FirstOrDefault().Dump();

    var output = results.FirstOrDefault().AsEnumerable<Single>().ToList();
    var str = "0123456789+-×÷=？";
    int maxIndex = 0;
    Single val = Single.MinValue;
    var lastRow = 0;
    var sb = new StringBuilder();
    for (int i = 0; i < output.Count; i++)
    {  //Console.WriteLine(output[i]);
        var row = 0;
        if (i > 0)
        {
            row = i / 16;
        }
        if (lastRow != row)
        {
            val = Single.MinValue;
            sb.Append(str[maxIndex]);
            maxIndex = -1;
            lastRow = row;
        }
        if (output[i] > val)
        {
            val = output[i];
            //Console.WriteLine(i-16*row);
            maxIndex = i - 16 * row;
        }
        if (i == output.Count - 1)
        {
            sb.Append(str[maxIndex]);
        }
    }

     return sb.ToString();
}

void ReadFile(Tensor<float> input)
{
    using var fs = File.Open(@"test\s.txt", FileMode.Open);
    using var reader = new StreamReader(fs);
    var str = reader.ReadToEnd();
    var stack = new Stack<char>();
    var channelIndex = 0;
    int windex = 0;
    int hindex = 0;
    var sIndex = 0;
    var span = str.AsSpan();
    for (int i = 0; i < span.Length; i++)
    {
        if (span[i] == '[')
        {
            stack.Push('[');
            sIndex = i + 1;
        }
        else if (span[i] == ']')
        {

            //span.Slice(sIndex,i-sIndex).ToString().Dump();
            if (sIndex != i)
            {
                var len = i - sIndex;
                var value = float.Parse(span.Slice(sIndex, len));
                input[0, channelIndex, windex, hindex] = value;
                hindex++;
                sIndex = i + 1;
            }
            stack.Pop();
            if (stack.Count() == 3)
            {
                windex++;
                hindex = 0;
                sIndex = i + 1;
            }
            else if (stack.Count() == 2)
            {
                channelIndex++;
                windex = 0;
                hindex = 0;
                sIndex = i + 1;
            }
            else if (stack.Count == 1)
            {
                break;
            }

        }
        else if (span[i] == ',')
        {
            if (sIndex != i)
            {
                //span.Slice(sIndex,i-sIndex).ToString().Dump();
                var len = i - sIndex;
                var value = float.Parse(span.Slice(sIndex, len));
                input[0, channelIndex, windex, hindex] = value;
                hindex++;
            }
            sIndex = i + 1;

        }
    }
}