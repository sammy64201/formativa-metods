/* USER CODE BEGIN Header */

/*
 * ADC2_TIMER_DAC
 * Referencia tomada de un perfil discreto (perfil_pos[250]) que simula una lectura ADC (0..2047 ≈ 0..1.65 V).
 * Retroalimentación tomada del ADC1_IN1 (PA1). Salida por DAC CH1 (PA4).
 *
 * Basado en ejemplos ADC_IT y TIM_IT. Modificado para usar perfil como referencia.
 */

/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <stdio.h>
#include <string.h>
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
/* --- Escalas hardware --- */
#define VREF                3.3f
#define DAC_COUNTS_MAX      4095.0f
#define ADC_COUNTS_MAX      4095.0f

/* Escala de la referencia proveniente del perfil (0..2047 counts).
   2047/4095*3.3 ≈ 1.65 V. Se puede ajustar con REF_SCALE y REF_OFFSET. */
#define REF_SCALE           1.0f     // multiplicador de amplitud
#define REF_OFFSET          0.0f     // offset en volts después de escalar

/* Control de avance del perfil:
   Con Fs=1000 Hz y STEP_DIV=10 → 250 puntos * 10 ms = 2.5 s por ciclo */
#define STEP_DIV            10       // bajar a 8 para ~2.0 s, subir para más lento

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
ADC_HandleTypeDef hadc1;
DAC_HandleTypeDef hdac;
TIM_HandleTypeDef htim10;

/* USER CODE BEGIN PV */
uint16_t adc_feedback; // almacena la conversión del ADC (retroalimentación)

/* FRECUENCIA DE MUESTREO */
uint16_t Fs = 1000; // Hz
uint32_t timer_clk;
uint32_t pclk;

/* --------- Perfil de referencia ---------- */
/* Perfil de movimiento típico de un ventilador (0..2047 counts) */
const uint16_t perfil_pos[250] =
{   0,    0,    8,   16,   32,   48,   64,   80,  104,  120,
  144,  169,  201,  225,  249,  281,  305,  337,  369,  401,
  433,  458,  490,  530,  562,  594,  626,  658,  690,  730,
  763,  795,  835,  867,  899,  939,  971, 1003, 1044, 1076,
 1108, 1148, 1180, 1212, 1252, 1284, 1317, 1357, 1389, 1421,
 1453, 1485, 1517, 1557, 1589, 1614, 1646, 1678, 1710, 1742,
 1766, 1798, 1822, 1846, 1878, 1903, 1927, 1943, 1967, 1983,
 1999, 2015, 2031, 2039, 2047, 2047, 2047, 2047, 2047, 2047,
 2047, 2047, 2047, 2047, 2047, 2047, 2047, 2047, 2047, 2047,
 2047, 2047, 2047, 2047, 2047, 2047, 2047, 2047, 2047, 2047,
 2047, 2047, 2047, 2047, 2047, 2047, 2047, 2047, 2047, 2047,
 2047, 2047, 2047, 2047, 2047, 2047, 2047, 2047, 2047, 2047,
 2047, 2047, 2047, 2047, 2047, 2047, 2047, 2039, 2031, 2023,
 2007, 1991, 1975, 1959, 1935, 1911, 1886, 1862, 1838, 1806,
 1782, 1750, 1718, 1686, 1654, 1622, 1589, 1549, 1517, 1477,
 1445, 1405, 1373, 1333, 1300, 1260, 1220, 1188, 1148, 1108,
 1076, 1036,  995,  963,  923,  891,  851,  819,  779,  747,
  714,  674,  642,  610,  578,  546,  514,  490,  458,  433,
  401,  377,  353,  329,  305,  289,  265,  249,  233,  217,
  201,  185,  177,  161,  153,  136,  128,  120,  104,   96,
   88,   80,   72,   64,   56,   48,   48,   40,   32,   24,
   24,   16,   16,    8,    8,    8,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0};
/* ---------------------------------------- */

#define PERF_LEN (sizeof(perfil_pos)/sizeof(perfil_pos[0]))

/* Variables PID */
volatile float r_k;     // referencia en V
volatile float y_k;     // retroalimentación en V
volatile float e_k;
volatile float E_k;
volatile float e_D;
volatile float u_k;

volatile float e_k_1 = 0;
volatile float E_k_1 = 0;

float kP = 8.3445f;
float kI = 24.3188f;
float kD = 0.20545f;

float dt;
float kI_ast;
float kD_ast;

/* Avance del perfil */
static uint16_t profile_idx = 0;
static uint16_t step_div_counter = 0;

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_ADC1_Init(void);
static void MX_TIM10_Init(void);
static void MX_DAC_Init(void);
/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

void HAL_TIM_PeriodElapsedCallback(TIM_HandleTypeDef *htim)
{
  if (htim->Instance != TIM10) return;

  /* --- Leer retroalimentación (ADC1_IN1 / PA1) --- */
  HAL_ADC_Start(&hadc1);
  HAL_ADC_PollForConversion(&hadc1, HAL_MAX_DELAY);
  adc_feedback = HAL_ADC_GetValue(&hadc1);
  y_k = ((float)adc_feedback * VREF) / ADC_COUNTS_MAX; // 0..3.3 V

  /* --- Obtener referencia desde el perfil (simula ADC 0..2047 ≈ 0..1.65 V) --- */
  float ref_counts = (float)perfil_pos[profile_idx];     // 0..2047
  r_k = (ref_counts * VREF) / ADC_COUNTS_MAX;            // convierte a volts (≈0..1.65)
  r_k = r_k * REF_SCALE + REF_OFFSET;                    // aplicar escala/offset si se desea
  if (r_k < 0.0f) r_k = 0.0f;
  if (r_k > VREF) r_k = VREF;

  /* --- PID incremental (posición) --- */
  e_k = r_k - y_k;
  E_k = E_k_1 + e_k;
  e_D = e_k - e_k_1;

  u_k = kP*e_k + kI_ast*E_k + kD_ast*e_D;               // salida en volts idealmente

  /* Convertir u_k (V) a cuentas DAC (0..4095) */
  float u_counts = (u_k * DAC_COUNTS_MAX) / VREF;

  if (u_counts > DAC_COUNTS_MAX) u_counts = DAC_COUNTS_MAX;
  if (u_counts < 0.0f)           u_counts = 0.0f;

  HAL_DAC_SetValue(&hdac, DAC_CHANNEL_1, DAC_ALIGN_12B_R, (uint16_t)u_counts);

  /* Actualizar estados PID */
  e_k_1 = e_k;
  E_k_1 = E_k;

  /* Avanzar el perfil cada STEP_DIV interrupciones */
  step_div_counter++;
  if (step_div_counter >= STEP_DIV) {
    step_div_counter = 0;
    profile_idx++;
    if (profile_idx >= PERF_LEN) profile_idx = 0; // repetir ciclo
  }
}

/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{
  HAL_Init();
  SystemClock_Config();

  MX_GPIO_Init();
  MX_ADC1_Init();   // ahora sólo IN1
  MX_TIM10_Init();
  MX_DAC_Init();

  /* USER CODE BEGIN 2 */
  HAL_DAC_Start(&hdac, DAC_CHANNEL_1);
  HAL_TIM_Base_Start_IT(&htim10);

  dt = 1.0f / (float)Fs;
  kI_ast = kI * dt;      // discretización de integral
  kD_ast = kD / dt;      // discretización de derivada
  /* USER CODE END 2 */

  /* Infinite loop */
  while (1) {
    /* todo ocurre en la ISR del TIM10 */
  }
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE3);

  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSI;
  RCC_OscInitStruct.PLL.PLLM = 16;
  RCC_OscInitStruct.PLL.PLLN = 336;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV4;
  RCC_OscInitStruct.PLL.PLLQ = 2;
  RCC_OscInitStruct.PLL.PLLR = 2;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK) { Error_Handler(); }

  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV2;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_2) != HAL_OK) { Error_Handler(); }
}

/**
  * @brief ADC1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_ADC1_Init(void)
{
  ADC_ChannelConfTypeDef sConfig = {0};

  hadc1.Instance = ADC1;
  hadc1.Init.ClockPrescaler        = ADC_CLOCK_SYNC_PCLK_DIV4;
  hadc1.Init.Resolution            = ADC_RESOLUTION_12B;
  hadc1.Init.ScanConvMode          = DISABLE;            // AHORA: 1 sola conversión
  hadc1.Init.ContinuousConvMode    = DISABLE;
  hadc1.Init.DiscontinuousConvMode = DISABLE;
  hadc1.Init.ExternalTrigConvEdge  = ADC_EXTERNALTRIGCONVEDGE_NONE;
  hadc1.Init.ExternalTrigConv      = ADC_SOFTWARE_START;
  hadc1.Init.DataAlign             = ADC_DATAALIGN_RIGHT;
  hadc1.Init.NbrOfConversion       = 1;                  // AHORA: 1 canal
  hadc1.Init.DMAContinuousRequests = DISABLE;
  hadc1.Init.EOCSelection          = ADC_EOC_SINGLE_CONV;
  if (HAL_ADC_Init(&hadc1) != HAL_OK) { Error_Handler(); }

  /* ÚNICO canal: ADC_CHANNEL_1 (PA1) para retroalimentación */
  sConfig.Channel      = ADC_CHANNEL_1;
  sConfig.Rank         = 1;
  sConfig.SamplingTime = ADC_SAMPLETIME_84CYCLES;
  if (HAL_ADC_ConfigChannel(&hadc1, &sConfig) != HAL_OK) { Error_Handler(); }
}

/**
  * @brief DAC Initialization Function
  * @param None
  * @retval None
  */
static void MX_DAC_Init(void)
{
  DAC_ChannelConfTypeDef sConfig = {0};

  hdac.Instance = DAC;
  if (HAL_DAC_Init(&hdac) != HAL_OK) { Error_Handler(); }

  sConfig.DAC_Trigger = DAC_TRIGGER_NONE;
  sConfig.DAC_OutputBuffer = DAC_OUTPUTBUFFER_ENABLE;
  if (HAL_DAC_ConfigChannel(&hdac, &sConfig, DAC_CHANNEL_1) != HAL_OK) { Error_Handler(); }
}

/**
  * @brief TIM10 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM10_Init(void)
{
  /* calcular timer_clk para TIM10 */
  pclk = HAL_RCC_GetPCLK2Freq();
  if ((RCC->CFGR & RCC_CFGR_PPRE2) != RCC_CFGR_PPRE2_DIV1) {
    timer_clk = pclk * 2;
  } else {
    timer_clk = pclk;
  }

  htim10.Instance = TIM10;
  htim10.Init.Prescaler = 8399; // 84MHz/(8399+1)=10kHz base
  htim10.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim10.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim10.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;

  /* Periodo para obtener Fs deseada */
  htim10.Init.Period = ((timer_clk / (htim10.Init.Prescaler + 1)) / Fs) - 1;

  if (HAL_TIM_Base_Init(&htim10) != HAL_OK) { Error_Handler(); }
}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};

  __HAL_RCC_GPIOC_CLK_ENABLE();
  __HAL_RCC_GPIOH_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();

  HAL_GPIO_WritePin(LD2_GPIO_Port, LD2_Pin, GPIO_PIN_RESET);

  GPIO_InitStruct.Pin = B1_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_IT_FALLING;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(B1_GPIO_Port, &GPIO_InitStruct);

  GPIO_InitStruct.Pin = LD2_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(LD2_GPIO_Port, &GPIO_InitStruct);
}

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  __disable_irq();
  while (1) { }
}

#ifdef USE_FULL_ASSERT
void assert_failed(uint8_t *file, uint32_t line)
{
  (void)file; (void)line;
}
#endif /* USE_FULL_ASSERT */
