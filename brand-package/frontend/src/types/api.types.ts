/* ============================================
   USER TYPES
   ============================================ */

export interface User {
  id: string;
  email: string;
  username?: string;
  created_at: string;
  updated_at?: string;
  generation_count: number;
  generation_limit: number;
  is_premium: boolean;
}

export interface AuthResponse {
  success: boolean;
  user: User;
  access_token: string;
  token_type: string;
  expires_in: number;
}

/* ============================================
   GENERATION REQUEST TYPES
   ============================================ */

export interface GeneratePackageRequest {
  description: string;
  business_name?: string;
  industry?: string;
  style_preferences?: StylePreferences;
  include_services?: string[];
}

export interface StylePreferences {
  name_style?: string;
  logo_style?: string;
  color_theme?: string;
  tagline_tone?: string;
}

/* ============================================
   NAME GENERATION TYPES
   ============================================ */

export interface NameOption {
  id: string;
  name: string;
  reasoning: string;
  score: number;
  style?: string;
  memorability?: number;
  pronounceability?: number;
  uniqueness?: number;
  domain_available?: boolean;
}

/* ============================================
   LOGO GENERATION TYPES
   ============================================ */

export interface LogoOption {
  id: string;
  concept_name: string;
  description: string;
  style: string;
  colors: string[];
  rationale: string;
  urls: LogoUrls;
  metadata?: LogoMetadata;
}

export interface LogoUrls {
  png: string;
  jpg: string;
  svg?: string;
}

export interface LogoMetadata {
  width: number;
  height: number;
  format: string;
  size_bytes: number;
  created_at: string;
}

/* ============================================
   COLOR PALETTE TYPES
   ============================================ */

export interface ColorPalette {
  id: string;
  name: string;
  description: string;
  theme?: string;
  colors: ColorInfo[];
  primary_color?: ColorInfo;
  secondary_color?: ColorInfo;
  accent_color?: ColorInfo;
  psychology?: string;
  use_cases: string[];
  contrasts: Record<string, number>;
  accessibility: AccessibilityInfo;
}

export interface ColorInfo {
  hex: string;
  rgb: string;
  hsl?: string;
  name: string;
  role: string;
}

export interface AccessibilityInfo {
  wcag_aa_normal: boolean;
  wcag_aa_large: boolean;
  wcag_aaa_normal: boolean;
  wcag_aaa_large: boolean;
  contrast_ratios?: Record<string, number>;
}

/* ============================================
   TAGLINE TYPES
   ============================================ */

export interface TaglineOption {
  id: string;
  text: string;
  tone: string;
  style?: string;
  reasoning: string;
  target_emotion?: string;
  call_to_action: boolean;
  word_count: number;
  character_count: number;
  readability: string;
  use_cases: string[];
}

/* ============================================
   DOMAIN CHECK TYPES
   ============================================ */

export interface DomainResult {
  domain: string;
  available: boolean;
  status: string;
  price?: string;
  currency?: string;
  registrar?: string;
  registrar_link?: string;
  checked_at: string;
  method: string;
  alternatives?: string[];
}

/* ============================================
   PACKAGE RESPONSE TYPE
   ============================================ */

export interface GeneratePackageResponse {
  success: boolean;
  project_id: string;
  generation_id: string;
  business_name: string;
  names?: NameOption[];
  domains?: DomainResult[];
  logos?: LogoOption[];
  color_palettes?: ColorPalette[];
  taglines?: TaglineOption[];
  summary: PackageSummary;
  errors: GenerationError[];
  total_cost: number;
  created_at: string;
}

export interface PackageSummary {
  business_name: string;
  components_generated: string[];
  recommendations: string[];
  featured_tagline?: string;
  primary_color?: string;
  best_domain?: string;
}

export interface GenerationError {
  service: string;
  error: string;
  timestamp?: string;
}

/* ============================================
   PROJECT TYPES
   ============================================ */

export interface Project {
  id: string;
  user_id: string;
  name: string;
  description: string;
  industry?: string;
  status: ProjectStatus;
  created_at: string;
  updated_at: string;
  generations?: Generation[];
}

export enum ProjectStatus {
  DRAFT = 'draft',
  IN_PROGRESS = 'in_progress',
  COMPLETED = 'completed',
  ARCHIVED = 'archived'
}

export interface Generation {
  id: string;
  project_id: string;
  user_id: string;
  generation_type: GenerationType;
  input_data: Record<string, any>;
  output_data: Record<string, any>;
  status: GenerationStatus;
  error_message?: string;
  created_at: string;
}

export enum GenerationType {
  NAME = 'name',
  LOGO = 'logo',
  COLOR = 'color',
  TAGLINE = 'tagline',
  DOMAIN = 'domain',
  PACKAGE = 'package'
}

export enum GenerationStatus {
  PENDING = 'pending',
  IN_PROGRESS = 'in_progress',
  SUCCESS = 'success',
  FAILED = 'failed'
}

/* ============================================
   ASSET TYPES
   ============================================ */

export interface Asset {
  id: string;
  generation_id: string;
  project_id: string;
  asset_type: AssetType;
  file_url: string;
  file_format: string;
  file_size: number;
  metadata?: Record<string, any>;
  created_at: string;
}

export enum AssetType {
  LOGO = 'logo',
  IMAGE = 'image',
  DOCUMENT = 'document',
  PALETTE = 'palette'
}

/* ============================================
   ERROR TYPES
   ============================================ */

export interface ApiError {
  error: string;
  message: string;
  details?: any;
  code?: string;
  status?: number;
}

/* ============================================
   PAGINATION TYPES
   ============================================ */

export interface PaginatedResponse<T> {
  data: T[];
  pagination: PaginationInfo;
}

export interface PaginationInfo {
  page: number;
  limit: number;
  total: number;
  total_pages: number;
  has_next: boolean;
  has_prev: boolean;
}

/* ============================================
   STREAMING TYPES
   ============================================ */

export interface StreamChunk {
  type: 'progress' | 'partial' | 'complete' | 'error';
  service?: string;
  data?: any;
  message?: string;
  progress?: number;
  timestamp: string;
}

/* ============================================
   HEALTH CHECK TYPE
   ============================================ */

export interface HealthCheckResponse {
  status: 'healthy' | 'unhealthy';
  timestamp: string;
  version?: string;
  services?: Record<string, ServiceHealth>;
}

export interface ServiceHealth {
  status: 'up' | 'down';
  latency_ms?: number;
  last_check?: string;
}