using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

using System.Diagnostics;
using System.Runtime.InteropServices;


namespace EdgeAI_Zonemaker
{
    public partial class Form1 : Form
    {
        // 1. 모든 윈도우 순회
        [DllImport("user32.dll")]
        private static extern bool EnumWindows(EnumWindowsProc enumProc, IntPtr lParam);

        public delegate bool EnumWindowsProc(IntPtr hWnd, IntPtr lParam);

        // 2. 윈도우 클래스 이름 가져오기
        [DllImport("user32.dll", SetLastError = true)]
        private static extern int GetWindowText(IntPtr hWnd, StringBuilder lpString, int nMaxCount);

        [DllImport("user32.dll", SetLastError = true)]
        private static extern int GetWindowTextLength(IntPtr hWnd);

        //3. 창이 보이는지 확인
        [DllImport("user32.dll")]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static extern bool IsWindowVisible(IntPtr hWnd);

        // 4. 프로세스 ID 가져오기
        [DllImport("user32.dll")]
        private static extern uint GetWindowThreadProcessId(IntPtr hWnd, out uint IpdwProcessId);

        // 5. WinAPI를 사용하여 현재 화면에 있는 창들을 가져오기 및 이동
        [DllImport("user32.dll")]
        private static extern bool SetWindowPos(IntPtr hWnd, IntPtr hWndInsertAfter, int X, int Y, int cx, int cy, uint uFlags);

        [DllImport("user32.dll")]
        static extern IntPtr GetForegroundWindow();

        // 상수 정의(선언)
        private const uint SWP_SHOWWINDOW = 0x0040;
        private static readonly IntPtr HWND_TOP = new IntPtr(0);

        // 무시할 프로세스 이름 리스트를 전역으로 선언 (Form1 클래스 안에 선언)
        string[] ignoreList = new string[]
        {
            "ApplicationFrameHost",       // UWP 앱을 감싸는 호스트 껍데기
            "SystemSettings",             // Windows 설정 창
            "TextInputHost",              // 가상 키보드, IME 입력 처리 관련
            //"ShellExperienceHost",        // 알림 센터, 타일 UI 등 Windows 쉘
            //"StartMenuExperienceHost",    // 시작 메뉴 UI
            //"SearchApp"                   // Windows 검색 UI
        };

        // 6. 현재 화면에 있는 창들을 가져오기(창 목록 수집)
        private List<IntPtr> GetTopWindows(int count)
        {
            /*
            List<IntPtr> windows = new List<IntPtr>();
            IntPtr myHandle = this.Handle;

            EnumWindows((hWnd, lParam) =>
            {
                
                // 내 창은 제외
                if (hWnd == myHandle || !IsWindowVisible(hWnd))
                    return true;

                // 프로세스 이름 가져오기
                GetWindowThreadProcessId(hWnd, out uint pid);
                try
                {
                    Process proc = Process.GetProcessById((int)pid);
                    if (ignoreList.Contains(proc.ProcessName))
                        return true; // 건너뛰기
                }
                catch
                {
                    return true; // 예외 시 건너뛰기
                }

                windows.Add(hWnd);
                return true;

            }, IntPtr.Zero);

            return windows.Take(count).ToList(); */
            
            List<IntPtr> windows = new List<IntPtr>();
            IntPtr myHandle = this.Handle; // 현재 내 창(폼)의 핸들

            EnumWindows((hWnd, lParam) =>
            {
                if (!IsWindowVisible(hWnd)) return true;

                if (hWnd == myHandle) return true; // 내 창은 제외

                StringBuilder buffer = new StringBuilder(256);
                GetWindowText(hWnd, buffer, buffer.Capacity);
                if (!string.IsNullOrWhiteSpace(buffer.ToString()))
                {
                    windows.Add(hWnd);
                }

                return true;
            }, IntPtr.Zero);

            // 최신 순 정렬은 어려우므로 일단 최근 앞쪽만 잘라서 사용
            return windows.Take(count).ToList(); 
        }

        private bool isLiveUpdateEnabled = true;

        // 콤보박스 값 save로 저장했을 때 받아주는 변수.
        private int selectedSplitCount = 0; // 기본값은 0. (선택되지 않음)

        public Form1()
        {
            InitializeComponent();
        }

        void ArrangeWindowsFromCombo()
        {
            if (comboBox1.SelectedItem == null) return;

            int count = int.Parse(comboBox1.SelectedItem.ToString());
            List<IntPtr> windows = GetTopWindows(count);
            if (windows.Count < count) return;

            int screenWidth = Screen.PrimaryScreen.Bounds.Width;
            int screenHeight = Screen.PrimaryScreen.Bounds.Height;

            if (count == 2)
            {
                int w = screenWidth / 2;
                for (int i = 0; i < 2; i++)
                {
                    SetWindowPos(windows[i], HWND_TOP, i * w, 0, w, screenHeight, SWP_SHOWWINDOW);
                }
            }
            else if (count == 3)
            {
                int leftWidth = screenWidth / 2;
                int rightWidth = screenWidth - leftWidth;
                int rightHeight = screenHeight / 2;

                // 창 1: 좌측 전체
                SetWindowPos(windows[0], HWND_TOP, 0, 0, leftWidth, screenHeight, SWP_SHOWWINDOW);

                // 창 2: 우측 상단
                SetWindowPos(windows[1], HWND_TOP, leftWidth, 0, rightWidth, rightHeight, SWP_SHOWWINDOW);

                // 창 3: 우측 하단
                SetWindowPos(windows[2], HWND_TOP, leftWidth, rightHeight, rightWidth, screenHeight - rightHeight, SWP_SHOWWINDOW);
            }
            else if (count == 4)
            {
                int colWidth = screenWidth / 3;
                int rowHeight = screenHeight / 3;

                // 비율 1:2:2:4 → 9등분 기준

                // 창 1: 좌상단 (1칸)
                SetWindowPos(windows[0], HWND_TOP, 0, 0, colWidth, rowHeight, SWP_SHOWWINDOW);

                // 창 2: 좌하단 (2칸)
                SetWindowPos(windows[1], HWND_TOP, 0, rowHeight, colWidth, screenHeight - rowHeight, SWP_SHOWWINDOW);

                // 창 3: 우상단 (2칸)
                SetWindowPos(windows[2], HWND_TOP, colWidth, 0, screenWidth - colWidth, rowHeight, SWP_SHOWWINDOW);

                // 창 4: 우하단 (4칸)
                SetWindowPos(windows[3], HWND_TOP, colWidth, rowHeight, screenWidth - colWidth, screenHeight - rowHeight, SWP_SHOWWINDOW);
            }
        }

        private void btnArrange_Click(object sender, EventArgs e)
        {
            ArrangeWindowsFromCombo();
        }

        private void ListOpenWindows()
        {
            List<string> windowTitles = new List<string>();

            EnumWindows(delegate (IntPtr hWnd, IntPtr lParam)
            {
                if (!IsWindowVisible(hWnd))
                    return true;

                int length = GetWindowTextLength(hWnd);
                if (length == 0)
                    return true;

                StringBuilder sb = new StringBuilder(length + 1);
                GetWindowText(hWnd, sb, sb.Capacity);

                uint processId;
                GetWindowThreadProcessId(hWnd, out processId);

                try
                {
                    Process proc = Process.GetProcessById((int)processId);
                    string title = sb.ToString();
                    windowTitles.Add($"{proc.ProcessName} - {title}");
                }
                catch { }

                return true;
            }, IntPtr.Zero);

            // 목록을 출력
            string msg = "실행 중인 창 목록:\n" + string.Join("\n", windowTitles);
            MessageBox.Show(msg);
        }

        private void RefreshWindowList()
        {
            listBox1.Items.Clear(); // 목록 초기화

            EnumWindows(delegate (IntPtr hWnd, IntPtr lParam)
            {
                if (!IsWindowVisible(hWnd))
                    return true;

                int length = GetWindowTextLength(hWnd);
                if (length == 0)
                    return true;

                StringBuilder sb = new StringBuilder(length + 1);
                GetWindowText(hWnd, sb, sb.Capacity);

                uint processId;
                GetWindowThreadProcessId(hWnd, out processId);

                try
                {
                    Process proc = Process.GetProcessById((int)processId);
                    string title = sb.ToString();
                    listBox1.Items.Add($"{proc.ProcessName} - {title}");
                }
                catch { }

                return true;
            }, IntPtr.Zero);
        }

        private void button1_Click(object sender, EventArgs e)
        {
            if (isLiveUpdateEnabled)
            {
                MessageBox.Show("Start 버튼을 눌러 Zonemaker AI를 실행한 후, 조작해주세요.", "조작불가", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                return;
            }

            try
            {
                ArrangeWindowsFromCombo();
                selectedSplitCount = int.Parse(comboBox1.SelectedItem.ToString()); // 리팩토링 필요한 부분(수동과 자동을 굳이 달리 할 필요 없으니). 버튼 누르면 받아주는 변수에 저장됨.
            }
            catch
            {
                MessageBox.Show("예상치 못한 오류 발생");
            }
        }

        private void pictureBox1_Click(object sender, EventArgs e)
        {

        }

        private void checkBox1_CheckedChanged(object sender, EventArgs e)
        {
            isLiveUpdateEnabled = !isLiveUpdateEnabled;

            // 버튼 텍스트 변경
            checkBox1.Text = isLiveUpdateEnabled ? "Start" : "Stop";

            if (checkBox1.Checked)
            {
                MessageBox.Show("ZonemakerAI is now Running!");
            }
            else
            {
                MessageBox.Show("ZonemakerAI Stopped. Press 'start' to run.");
            }
        }

        private void label2_Click(object sender, EventArgs e)
        {

        }

        private void Form1_Load(object sender, EventArgs e)
        {
            comboBox1.Items.AddRange(new object[] {"1", "2", "3", "4" });
            comboBox1.SelectedIndex = 1; // Set default selection to the first item 
            RefreshWindowList();
        }

        private void comboBox1_SelectedIndexChanged(object sender, EventArgs e)
        {

        }

        private void label1_Click(object sender, EventArgs e)
        {

        }

        private void label4_Click(object sender, EventArgs e)
        {

        }

        private List<IntPtr> previousWindows = new List<IntPtr>();


        private void ArrangeWindows()
        {
            if (comboBox1.SelectedItem == null) return;

            if (selectedSplitCount == 0) return; // 안해줘도 어차피 조건문에는 2,3,4밖에 없어서 괜찮긴 한데, 일단 안전하게 예외처리.

            //int count = int.Parse(comboBox1.SelectedItem.ToString());
            int count = selectedSplitCount; // 기존에 콤보박스에서 바로 받아주는 방식에서, save 버튼눌러 저장된 값을 받아주는 방식으로.

            List<IntPtr> windows = GetTopWindows(count);
            if (windows.Count < count) return;

            int screenWidth = Screen.PrimaryScreen.Bounds.Width;
            int screenHeight = Screen.PrimaryScreen.Bounds.Height;

            // 분할 로직은 이전과 동일
            if (count == 2)
            {
                int w = screenWidth / 2;
                for (int i = 0; i < 2; i++)
                {
                    SetWindowPos(windows[i], HWND_TOP, i * w, 0, w, screenHeight, SWP_SHOWWINDOW);
                }
            }
            else if (count == 3)
            {
                int leftWidth = screenWidth / 2;
                int rightWidth = screenWidth - leftWidth;
                int rightHeight = screenHeight / 2;

                // 창 1: 좌측 전체
                SetWindowPos(windows[0], HWND_TOP, 0, 0, leftWidth, screenHeight, SWP_SHOWWINDOW);

                // 창 2: 우측 상단
                SetWindowPos(windows[1], HWND_TOP, leftWidth, 0, rightWidth, rightHeight, SWP_SHOWWINDOW);

                // 창 3: 우측 하단
                SetWindowPos(windows[2], HWND_TOP, leftWidth, rightHeight, rightWidth, screenHeight - rightHeight, SWP_SHOWWINDOW);
            }
            else if (count == 4)
            {
                int colWidth = screenWidth / 3;
                int rowHeight = screenHeight / 3;

                // 비율 1:2:2:4 → 9등분 기준

                // 창 1: 좌상단 (1칸)
                SetWindowPos(windows[0], HWND_TOP, 0, 0, colWidth, rowHeight, SWP_SHOWWINDOW);

                // 창 2: 좌하단 (2칸)
                SetWindowPos(windows[1], HWND_TOP, 0, rowHeight, colWidth, screenHeight - rowHeight, SWP_SHOWWINDOW);

                // 창 3: 우상단 (2칸)
                SetWindowPos(windows[2], HWND_TOP, colWidth, 0, screenWidth - colWidth, rowHeight, SWP_SHOWWINDOW);

                // 창 4: 우하단 (4칸)
                SetWindowPos(windows[3], HWND_TOP, colWidth, rowHeight, screenWidth - colWidth, screenHeight - rowHeight, SWP_SHOWWINDOW);
            }

            // 현재 상태 기억
            previousWindows = windows;
        }
        private void timer1_Tick(object sender, EventArgs e)
        {
            if (isLiveUpdateEnabled) return; // Stop 상태이면 아무것도 안함.

            RefreshWindowList();

            // 현재 창 목록이 바뀐 경우에만 Arrange 실행
            List<IntPtr> currentWindows = GetTopWindows(int.Parse(comboBox1.SelectedItem?.ToString() ?? "0"));

            if (!currentWindows.SequenceEqual(previousWindows))
            {
                ArrangeWindows(); // 자동 재배치
            }
        }
    }
}
