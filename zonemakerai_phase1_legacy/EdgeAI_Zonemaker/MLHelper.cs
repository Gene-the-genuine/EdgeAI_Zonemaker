using System.Threading.Tasks;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;

namespace EdgeAI_Zonemaker
{
    public static class MLHelper
    {
        private static InferenceSession mlSession;

        // 모델 초기화 (정적 생성자)
        static MLHelper()
        {
            string modelPath = System.IO.Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "priority_model.onnx");
            mlSession = new InferenceSession(modelPath);
        }

        // 예측 메서드
        public static float PredictWindowPriority(Dictionary<string, object> features)
        {
            var titleLength = Convert.ToSingle(features["title_length"]);
            var isFocused = (bool)features["is_focused"] ? 1f : 0f;
            var isFullscreen = (bool)features["is_fullscreen"] ? 1f : 0f;

            var inputTensor = new DenseTensor<float>(
                new[] { titleLength, isFocused, isFullscreen },
                new[] { 1, 3 }  // shape: 1x3
            );

            var inputs = new List<NamedOnnxValue>
            {
             NamedOnnxValue.CreateFromTensor("input", inputTensor)
            };
            /*
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("title_length", new DenseTensor<float>(new[] { Convert.ToSingle(features["title_length"]) }, new[] { 1, 1 })),
                NamedOnnxValue.CreateFromTensor("is_focused", new DenseTensor<float>(new[] { (bool)features["is_focused"] ? 1f : 0f }, new[] { 1, 1 })),
                NamedOnnxValue.CreateFromTensor("is_fullscreen", new DenseTensor<float>(new[] { (bool)features["is_fullscreen"] ? 1f : 0f }, new[] { 1, 1 })),
                // 필요 시 다른 feature도 추가 가능
            };
            */

            using var results = mlSession.Run(inputs);
            var score = results.First().AsEnumerable<float>().First();
            return score;
        }

        // 창 정렬 메서드
        public static List<IntPtr> SortWindowsByPriority(List<IntPtr> windows)
        {
            return windows.OrderByDescending(hWnd =>
            {
                var features = ExtractWindowFeatures(hWnd);
                return PredictWindowPriority(features);
            }).ToList();
        }

        // 창에서 feature 추출
        private static Dictionary<string, object> ExtractWindowFeatures(IntPtr hWnd)
        {
            // 제목 길이
            StringBuilder sb = new StringBuilder(256);
            GetWindowText(hWnd, sb, sb.Capacity);
            string title = sb.ToString();

            // 포커스 여부
            IntPtr foreground = GetForegroundWindow();
            bool isFocused = (foreground == hWnd);

            // 전체화면 여부 (간단한 추정)
            bool isFullscreen = IsWindowFullscreen(hWnd);

            return new Dictionary<string, object>
            {
                { "title_length", title.Length },
                { "is_focused", isFocused },
                { "is_fullscreen", isFullscreen }
            };
        }

        // 전체화면 추정 (예시)
        private static bool IsWindowFullscreen(IntPtr hWnd)
        {
            GetWindowRect(hWnd, out RECT rect);
            int screenWidth = GetSystemMetrics(0);
            int screenHeight = GetSystemMetrics(1);
            return rect.Left == 0 && rect.Top == 0 && rect.Right == screenWidth && rect.Bottom == screenHeight;
        }

        // Win32 API 선언
        [DllImport("user32.dll")]
        private static extern IntPtr GetForegroundWindow();

        [DllImport("user32.dll")]
        private static extern bool GetWindowRect(IntPtr hWnd, out RECT lpRect);

        [DllImport("user32.dll", CharSet = CharSet.Auto)]
        private static extern int GetWindowText(IntPtr hWnd, StringBuilder lpString, int nMaxCount);

        [DllImport("user32.dll")]
        private static extern int GetSystemMetrics(int nIndex);

        private struct RECT
        {
            public int Left, Top, Right, Bottom;
        }
    }
}