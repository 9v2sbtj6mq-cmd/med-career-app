const crypto = require("crypto");
const fs = require("fs");
const path = require("path");
const express = require("express");
const dotenv = require("dotenv");
const mammoth = require("mammoth");
const { tavily } = require("@tavily/core");
const { chromium } = require("playwright");
const NodeCache = require("node-cache");
const rateLimit = require("express-rate-limit");
const { Document, Packer, Paragraph, HeadingLevel, TextRun, AlignmentType } = require("docx");

dotenv.config();

const app = express();
function logAnalytics(event, details = {}) {
  const entry = {
    time: new Date().toISOString(),
    event,
    ...details
  };

  console.log("[analytics]", event);

  fs.appendFile(
    path.join(__dirname, "analytics.log"),
    JSON.stringify(entry) + "\n",
    () => {}
  );
}
app.use(express.json({ limit: "25mb" }));
app.use(express.static("."));

const apiLimiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: 120,
  standardHeaders: true,
  legacyHeaders: false,
  message: {
    error: "Too many requests. Please wait a few minutes and try again."
  }
});

app.use(apiLimiter);

app.use((req, res, next) => {
  const publicRoutes = ["/", "/index.html", "/jobs.html", "/auth"];

  if (publicRoutes.includes(req.path)) {
    return next();
  }

  const sessionToken = req.headers["x-session-token"] || req.headers["x-app-key"];

  if (!sessionToken || sessionToken !== APP_SESSION_TOKEN) {
    return res.status(401).json({ error: "Unauthorized request." });
  }

  next();
});

app.post("/auth", (req, res) => {
  const { accessCode } = req.body || {};

  if (!accessCode || accessCode !== APP_ACCESS_CODE) {
    logAnalytics("auth_failed", { ip: req.ip });
    return res.status(401).json({ error: "Invalid access code." });
  }

  logAnalytics("auth_success", { ip: req.ip });
  res.json({ token: APP_SESSION_TOKEN });
});

const GOOGLE_API_KEY = process.env.GOOGLE_API_KEY;
const TAVILY_API_KEY = process.env.TAVILY_API_KEY;
const FIRECRAWL_API_KEY = process.env.FIRECRAWL_API_KEY;
const GROQ_API_KEY = process.env.GROQ_API_KEY;
const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY;
const APP_ACCESS_CODE = process.env.APP_ACCESS_CODE || "med123";
const APP_SESSION_TOKEN = process.env.APP_SESSION_TOKEN || process.env.APP_ACCESS_KEY || "med-career-private-beta-2026";

const MODEL_FAST = process.env.GEMINI_MODEL_FAST || "gemini-2.5-flash-lite";
const MODEL_SMART = process.env.GEMINI_MODEL_SMART || "gemini-2.5-flash";
const GROQ_MODEL = process.env.GROQ_MODEL || "llama-3.1-8b-instant";
const OPENROUTER_MODEL = process.env.OPENROUTER_MODEL || "meta-llama/llama-3.1-8b-instruct:free";
const SCORING_PROVIDER = (process.env.SCORING_PROVIDER || "groq").toLowerCase();
const FIRECRAWL_TOP_N = Number(process.env.FIRECRAWL_TOP_N || 10);

const tavilyClient = tavily({ apiKey: TAVILY_API_KEY });
const searchCache = new NodeCache({ stdTTL: 900 });
const scoreCache = new NodeCache({ stdTTL: 1800 });
const geminiCache = new NodeCache({ stdTTL: 3600 });

async function askGemini(prompt, model = MODEL_SMART) {
  if (!GOOGLE_API_KEY) throw new Error("Missing GOOGLE_API_KEY in .env file.");

  const response = await fetch(
    `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${GOOGLE_API_KEY}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ contents: [{ parts: [{ text: prompt }] }] })
    }
  );

  const data = await response.json();
  if (!response.ok) throw new Error(data.error?.message || "Gemini API error");

  return data.candidates?.[0]?.content?.parts?.map(p => p.text).join("\n") || "";
}

async function askGroq(prompt, model = GROQ_MODEL) {
  if (!GROQ_API_KEY) throw new Error("Missing GROQ_API_KEY in .env file.");

  const response = await fetch("https://api.groq.com/openai/v1/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${GROQ_API_KEY}`
    },
    body: JSON.stringify({
      model,
      temperature: 0.2,
      messages: [
        {
          role: "system",
          content: "You are a strict JSON-only Australian medical recruitment scoring assistant. Return valid JSON only."
        },
        {
          role: "user",
          content: prompt
        }
      ]
    })
  });

  const data = await response.json();
  if (!response.ok) throw new Error(data.error?.message || "Groq API error");

  return data.choices?.[0]?.message?.content || "";
}

async function askOpenRouter(prompt, model = OPENROUTER_MODEL) {
  if (!OPENROUTER_API_KEY) throw new Error("Missing OPENROUTER_API_KEY in .env file.");

  const response = await fetch("https://openrouter.ai/api/v1/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${OPENROUTER_API_KEY}`,
      "HTTP-Referer": "http://localhost:3000",
      "X-Title": "Med Career App"
    },
    body: JSON.stringify({
      model,
      temperature: 0.2,
      messages: [
        {
          role: "system",
          content: "You are a strict JSON-only Australian medical recruitment scoring assistant. Return valid JSON only."
        },
        {
          role: "user",
          content: prompt
        }
      ]
    })
  });

  const data = await response.json();
  if (!response.ok) throw new Error(data.error?.message || "OpenRouter API error");

  return data.choices?.[0]?.message?.content || "";
}

async function askScoringModel(prompt) {
  try {
    if (SCORING_PROVIDER === "openrouter") {
      return await askOpenRouter(prompt);
    }

    if (SCORING_PROVIDER === "gemini") {
      return await askGemini(prompt, MODEL_FAST);
    }

    return await askGroq(prompt);
  } catch (primaryError) {
    console.warn(`${SCORING_PROVIDER} scoring failed. Falling back to Gemini fast model:`, primaryError.message);
    return askGemini(prompt, MODEL_FAST);
  }
}

async function scrapeJobWithFirecrawl(url) {
  if (!FIRECRAWL_API_KEY) return "";
  if (!url || !/^https?:\/\//i.test(url)) return "";

  try {
    const response = await fetch("https://api.firecrawl.dev/v1/scrape", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${FIRECRAWL_API_KEY}`
      },
      body: JSON.stringify({
        url,
        formats: ["markdown"],
        onlyMainContent: true,
        waitFor: 1000
      })
    });

    const data = await response.json();
    if (!response.ok) {
      console.warn("Firecrawl scrape failed:", data.error || data.message || response.statusText);
      return "";
    }

    return data.data?.markdown || data.markdown || "";
  } catch (error) {
    console.warn("Firecrawl scrape error:", error.message);
    return "";
  }
}

async function enrichTopJobsWithFirecrawl(jobs) {
  const topJobs = jobs.slice(0, FIRECRAWL_TOP_N);
  const remainingJobs = jobs.slice(FIRECRAWL_TOP_N);

  const enrichedTopJobs = await Promise.all(
    topJobs.map(async job => {
      const fullDescription = await scrapeJobWithFirecrawl(job.link);

      return addClosingDateInfo({
        ...job,
        fullDescription: fullDescription ? fullDescription.slice(0, 12000) : job.fullDescription || "",
        descriptionSource: fullDescription ? "firecrawl" : job.descriptionSource || "snippet"
      });
    })
  );

  return [...enrichedTopJobs, ...remainingJobs];
}

async function askGeminiCached(cacheKey, prompt, model = MODEL_SMART) {
  const cached = geminiCache.get(cacheKey);

  if (cached) {
    return cached;
  }

  const result = await askGemini(prompt, model);
  geminiCache.set(cacheKey, result);
  return result;
}

function extractJson(text) {
  const cleaned = text.replace(/```json/g, "").replace(/```/g, "").trim();
  const start = cleaned.indexOf("[");
  const end = cleaned.lastIndexOf("]");
  if (start === -1 || end === -1) return [];

  try {
    return JSON.parse(cleaned.slice(start, end + 1));
  } catch {
    return [];
  }
}

function parseScore(scoreText) {
  const match = String(scoreText || "").match(/Score:\s*([0-5](?:\.\d+)?)/i);
  return match ? Number(match[1]) : 0;
}

function buildScoreText(score) {
  if (!score) {
    return "Score not available. Try again or paste full job description.";
  }

  const breakdown = score.breakdown || {};
  const numericScore = Number(score.score || 0);

  return [
    `Score: ${numericScore}/5`,
    `Recommendation: ${score.recommendation || "Maybe"}`,
    `Confidence: ${score.confidence || "Medium"}`,
    `Apply readiness: ${score.applyReadiness || "Needs manual check"}`,
    "",
    "Breakdown:",
    `Registration: ${breakdown.registration ?? "?"}/1`,
    `Visa: ${breakdown.visa ?? "?"}/1`,
    `Level: ${breakdown.level ?? "?"}/1`,
    `Training value: ${breakdown.trainingValue ?? "?"}/1`,
    `Practical fit: ${breakdown.practicalFit ?? "?"}/1`,
    "",
    "Reason:",
    score.reason || "No reason provided.",
    "",
    "Warning:",
    score.warning || "No major warning identified from supplied information."
  ].join("\n");
}


function stableStringify(value) {
  if (value === null || value === undefined) return "";
  if (typeof value === "string") return value;
  if (typeof value !== "object") return String(value);

  if (Array.isArray(value)) {
    return `[${value.map(stableStringify).join(",")}]`;
  }

  return `{${Object.keys(value)
    .sort()
    .map(key => `${key}:${stableStringify(value[key])}`)
    .join(",")}}`;
}

function makeGeminiCacheKey(prefix, profile, job, model = "") {
  const raw = `${prefix}:${model}:${stableStringify(profile)}:${stableStringify(job)}`;
  return crypto.createHash("sha256").update(raw).digest("hex");
}


function profileToText(profile) {
  if (!profile) return "No profile provided.";

  if (typeof profile === "string") return profile;

  const selectedReferees = [1, 2, 3]
    .map(number => {
      if (!profile[`includeReferee${number}`]) return "";

      const name = profile[`referee${number}Name`] || "";
      const email = profile[`referee${number}Email`] || "";
      const phone = profile[`referee${number}Phone`] || "";
      const department = profile[`referee${number}Department`] || "";
      const hospital = profile[`referee${number}Hospital`] || "";

      if (!name && !email && !phone && !department && !hospital) return "";

      return [
        `Referee ${number}`,
        `Name: ${name || "[Add referee name]"}`,
        `Email: ${email || "[Add email]"}`,
        `Phone: ${phone || "[Add phone]"}`,
        `Department / Position: ${department || "[Add department / position]"}`,
        `Hospital / Health Service: ${hospital || "[Add hospital]"}`
      ].join("\n");
    })
    .filter(Boolean)
    .join("\n\n");

  return `
Visa: ${profile.visa || ""}
Needs Sponsorship: ${profile.needsSponsorship ? "Yes" : "No"}
AHPRA: ${profile.ahpra || ""}
Level: ${profile.jobLevel || profile.level || ""}
Preferred State: ${profile.stateFilter || "All states"}
Specialty Interest: ${profile.specialtyInterest || ""}
Willing to Relocate: ${profile.willingToRelocate ? "Yes" : "No"}

Name: ${profile.name || ""}
Email: ${profile.email || ""}
Phone: ${profile.phone || ""}
Location: ${profile.location || ""}
LinkedIn: ${profile.linkedin || ""}

CV Mode: ${profile.cvMode || "Use Structured Template"}

Existing CV:
${profile.baseCv || ""}

Structured CV Template:

Education:
${profile.education || ""}

Professional Summary:
${profile.cvSummary || ""}

Experience:
${profile.experience || profile.cvWorkHistory || ""}

Skills:
${profile.skills || profile.cvClinicalSkills || ""}

Courses / Certifications:
${profile.cvCourses || ""}

Audits / Research / Publications:
${profile.cvAudits || ""}

Extra Notes:
${profile.extraNotes || ""}

Referees:
${profile.cvReferees || ""}

Selected Referees for this application:
${profile.refereesOnRequest ? "Referees available on request" : (selectedReferees || "No referees selected or supplied.")}
`;
}

function buildSearchQueryFromProfile(profile) {
  if (!profile || typeof profile === "string") {
    return "resident medical officer RMO jobs Australia hospital doctor";
  }

  const level = String(profile.jobLevel || profile.level || "RMO").toLowerCase();
  const state = String(profile.stateFilter || "All states");
  const specialty = String(profile.specialtyInterest || "").trim();

  let base = "";

  if (level.includes("intern")) {
    base = "medical intern jobs Australia hospital doctor intern";
  } else if (level.includes("pho")) {
    base = "principal house officer PHO registrar jobs Australia hospital doctor unaccredited registrar";
  } else if (level.includes("registrar")) {
    base = "registrar unaccredited registrar medical officer jobs Australia hospital doctor";
  } else {
    base = "resident medical officer RMO hospital medical officer HMO junior medical officer JMO jobs Australia hospital doctor";
  }

  const location = state && state !== "All states" ? state : "";
  const sponsorship = profile.needsSponsorship
    ? "IMG friendly visa sponsorship supervision hospital doctor"
    : "";

  return [base, location, sponsorship, specialty]
    .filter(Boolean)
    .join(" ")
    .replace(/\s+/g, " ")
    .trim();
}


function buildMedicalContext(profile, job) {
  return `
Doctor Profile:
${profileToText(profile)}

Job Description:
${job}
`;
}

// Helper functions for safe download filenames
function safeFilenamePart(value = "", fallback = "Document") {
  return String(value || fallback)
    .trim()
    .replace(/[^a-z0-9]+/gi, "_")
    .replace(/^_+|_+$/g, "")
    .slice(0, 60) || fallback;
}

function extractFieldFromProfileText(profileText = "", fieldName = "") {
  const regex = new RegExp(`${fieldName}:\\s*(.*)`, "i");
  return String(profileText || "").match(regex)?.[1]?.trim() || "";
}

function extractJobTitleFromJobText(job = "") {
  const text = String(job || "");
  const titleMatch = text.match(/Title:\s*(.*)/i);
  if (titleMatch?.[1]) return titleMatch[1].trim();

  const selectedMatch = text.match(/Selected job:\s*(.*)/i);
  if (selectedMatch?.[1]) return selectedMatch[1].trim();

  return "Job";
}

function buildDownloadFilename(profile, job, documentType = "Document") {
  const profileText = profileToText(profile);
  const name = safeFilenamePart(extractFieldFromProfileText(profileText, "Name"), "Medical");
  const level = safeFilenamePart(extractFieldFromProfileText(profileText, "Level"), "Doctor");
  const jobTitle = safeFilenamePart(extractJobTitleFromJobText(job), "Job");
  const type = safeFilenamePart(documentType, "Document");

  return `${name}_${level}_${jobTitle}_${type}.docx`;
}


function cvTailoringInstructions() {
  return `
CV tailoring instructions:
- First detect CV Mode from the Doctor Profile, but do not treat the modes as completely separate.
- If CV Mode is "Use Pasted CV", treat Existing CV as the main source of truth. Preserve the candidate's real chronology, roles, qualifications, and clinical experience.
- If structured fields also contain useful information, use them to fill gaps or strengthen the pasted CV, but do not let structured fields override the pasted CV unless they clearly add missing details.
- If CV Mode is "Use Structured Template", build the CV mainly from the structured template fields.
- If both Existing CV and structured fields are supplied, intelligently merge both:
  • Existing CV controls dates, roles, qualifications, registrations, publications, and real chronology.
  • Structured fields can add missing professional summary, skills, courses, audits, extra notes, and referee preferences.
  • If there is a conflict between pasted CV and structured fields, prefer the pasted CV and use a placeholder/comment only if clarification is needed.
- Do not invent hospitals, dates, qualifications, registration status, visa status, procedures, audits, publications, referees, or achievements.
- If information is missing, include a clear placeholder such as [Add dates] or [Add hospital name] rather than making it up.
- Tailor the CV to the job description by prioritising matching clinical experience, registration/supervision fit, visa/sponsorship fit, ED/hospital/rotational experience, courses, audits, and skills.
- Reorder bullet points within sections so the most relevant material for this job appears first.
- Keep the tone polished, natural, human, and suitable for Australian hospital medical recruitment.
- Make the CV feel like a strong edited version of the candidate's real CV, not a newly invented AI document.
- Avoid generic AI phrases such as passionate, dynamic, fast-paced, cutting-edge, robust, leverage, and seamless.
- If unsure, prefer omission over guessing.
- Keep formatting clean with clear headings and concise bullet points.
`;
}

function mergedCvModeInstructions() {
  return `
Merged CV mode rules:
- Do not ignore structured fields just because an uploaded/pasted CV exists.
- Do not ignore the uploaded/pasted CV just because structured fields exist.
- Use uploaded/pasted CV as the factual backbone whenever available.
- Use structured fields as supporting evidence and gap-fillers.
- If the uploaded CV has old or overseas contact details but the profile form has updated Australian contact details, use the profile form contact details.
- If uploaded CV says referees are available on request and the profile form also says referees available on request, keep that wording.
- If structured fields are blank, do not mention them.
- If structured fields contain extra notes relevant to the job, integrate them naturally into the appropriate CV section.
- If both sources repeat the same information, include it once only.
`;
}

function jobCriteriaExtractionInstructions() {
  return `
Before writing the CV, extract the job's key selection criteria internally:
- Role level and specialty area
- Must-have clinical skills
- Preferred clinical skills
- Registration/AHPRA requirements
- Visa/sponsorship clues
- Location/practical fit
- Keywords likely used by recruiters or ATS systems

Then use those criteria to decide what to emphasise in the CV. Do not print this analysis unless specifically asked.
`;
}

function cvQualityInstructions() {
  return `
CV quality rules:
- Write strong, action-based clinical bullet points.
- Convert vague bullets into practical clinical statements while staying truthful.
- Match candidate experience directly to the job requirements.
- Remove or reduce weak, irrelevant, or repetitive content.
- Keep the most relevant experience near the top of each section.
- Do not contradict any information provided in the candidate profile or pasted CV.
- Use a targeted 4 to 5 line professional profile at the top.
- Make the CV sound like it was written by a careful human Australian medical applicant, not an AI tool.
- Prefer direct clinical statements over generic personality claims.
- Preserve the candidate's genuine experience and improve wording without changing the meaning.
`;
}

function humanWritingInstructions() {
  return `
Human writing style rules:
- Write like a careful, real medical applicant, not like an AI generator.
- Use simple, confident, professional language.
- Avoid inflated phrases such as passionate, dynamic, fast-paced, cutting-edge, robust, seamless, leverage, proven track record, highly motivated, and exceptional.
- Avoid over-selling. The tone should be credible, grounded, and clinically mature.
- Use specific clinical duties and real experience from the supplied CV rather than generic claims.
- Keep sentences varied. Do not make every bullet sound the same.
- Prefer practical hospital language: assessed, escalated, documented, coordinated, reviewed, assisted, performed, followed up, communicated, and contributed.
- Do not invent facts, Australian experience, AHPRA status, visa status, hospitals, referees, courses, audits, or procedures.
- If a detail is missing, use a simple placeholder rather than guessing.
`;
}

function topTierMedicalCvInstructions() {
  return `
Create a top-tier Australian hospital medical CV matched to the selected role level.

Role targeting:
- If Level is Intern: emphasise internship readiness, rotations, safe escalation, teamwork, documentation, basic procedures, and willingness to learn.
- If Level is RMO: emphasise independent ward work, ED/acute care exposure, procedural skills, admissions, discharge planning, escalation, multidisciplinary care, and reliability.
- If Level is PHO: emphasise registrar-readiness, leadership, supervision of junior staff, after-hours responsibility, procedural competence, acute decision-making, audits, teaching, and service contribution.

CV format:
- Use a professional Australian hospital CV structure.
- Start with name and contact placeholder if missing.
- Then Professional Profile.
- Then Registration / Visa / Work Rights.
- Then Key Clinical Skills.
- Then Employment History / Clinical Experience.
- Then Education.
- Then Courses / Certifications.
- Then Audits / Research / Publications.
- Then Teaching / Leadership if supplied or relevant.
- Then Referees.

Writing rules:
- Use bullet points beginning with "•" for skills, experience, audits, and achievements.
- Each bullet must start with a strong action verb where possible, such as Managed, Assessed, Coordinated, Performed, Assisted, Documented, Escalated, Led, Supported, or Contributed.
- Keep bullets concise, practical, and clinically relevant.
- Avoid long paragraphs in clinical experience sections.
- Avoid generic AI wording.
- Do not invent anything.
- Use placeholders such as [Add date], [Add hospital], [Add contact details], or [Add referee details] if needed.
- Tailor strongly to the job description and selected role level.
`;
}

function extractClosingDateFromText(text = "") {
  const source = String(text || "");
  const match = source.match(/(?:closing date|applications close|application close date|closes|closing)[:\s-]*([^\n|]+)/i);
  return match?.[1]?.trim() || "";
}

function getClosingStatus(closingDateText = "") {
  if (!closingDateText) return "No closing date found";

  const parsed = Date.parse(closingDateText);
  if (Number.isNaN(parsed)) return "Check closing date";

  const diffDays = Math.ceil((new Date(parsed) - new Date()) / (1000 * 60 * 60 * 24));
  if (diffDays < 0) return "Closed";
  if (diffDays <= 7) return "Closing soon";
  return "Open";
}

function getDaysUntilClosing(closingDateText = "") {
  if (!closingDateText) return null;

  const parsed = Date.parse(closingDateText);
  if (Number.isNaN(parsed)) return null;

  const today = new Date();
  today.setHours(0, 0, 0, 0);

  const closingDate = new Date(parsed);
  closingDate.setHours(0, 0, 0, 0);

  return Math.ceil((closingDate - today) / (1000 * 60 * 60 * 24));
}

function addClosingDateInfo(job) {
  const text = `${job.title || ""}\n${job.snippet || ""}\n${job.fullDescription || ""}`;
  const closingDate = extractClosingDateFromText(text);
  const finalClosingDate = job.closingDate || closingDate;
  const daysUntilClosing = getDaysUntilClosing(finalClosingDate);

  return {
    ...job,
    closingDate: finalClosingDate,
    closingStatus: job.closingStatus || getClosingStatus(finalClosingDate),
    daysUntilClosing,
    expiryLabel: daysUntilClosing === null
      ? "Closing date not found"
      : daysUntilClosing < 0
        ? "Expired"
        : daysUntilClosing === 0
          ? "Closes today"
          : daysUntilClosing === 1
            ? "1 day left"
            : `${daysUntilClosing} days left`
  };
}

function scoringFramework() {
  return `
Score each job out of 5 using this framework:

1. Registration fit /1
2. Visa fit /1
3. Level fit /1
4. Training value /1
5. Practical fit /1

Rules:
- Be conservative. Avoid false reassurance.
- A total score of 5/5 should mean the applicant is realistically a strong candidate, not just possibly eligible.
- If job is clearly too senior, such as consultant/staff specialist/director, score low for RMO/PHO users.
- If job is nursing/allied health, score 0 and recommend Skip.
- Use only the supplied job data unless the full job description is provided elsewhere in the request.

Strict registration scoring:
- Registration = 1/1 ONLY if the applicant already clearly meets the job's registration requirement OR the job explicitly accepts limited/provisional registration, supervised practice, IMGs, or registration-eligible applicants.
- Registration = 0.5/1 if the applicant is only eligible for registration but the job does not clearly mention IMG suitability, supervision, limited/provisional registration, or registration-eligible applicants.
- Registration = 0/1 if the role clearly requires general/specialist registration and the applicant does not currently hold that registration.
- Do not treat "eligible for registration" as the same as "currently registered".

Strict visa scoring:
- Visa = 1/1 ONLY if sponsorship, 482 sponsorship, employer nomination, IMG suitability, or international applicants are clearly supported or strongly implied.
- Visa = 0.5/1 if sponsorship is unclear but the role is in a public hospital, rural/regional area, area of need, workforce shortage setting, or IMG-friendly context where sponsorship may be possible.
- Visa = 0/1 if unrestricted work rights, PR, citizenship, or no sponsorship is required or strongly implied.
- If visa is Citizen or Have working rights, do not penalise lack of sponsorship.
- If visa is Requires sponsorship, do not give full visa marks unless sponsorship/IMG suitability is clear or strongly implied.

Apply readiness must be one of:
Ready to apply now / Needs visa clarity / Needs AHPRA/supervision clarity / Not suitable

Return valid JSON array only. Do not include markdown. Do not include explanation outside JSON.
[
  {
    "jobNumber": 1,
    "score": 4.2,
    "recommendation": "Apply",
    "confidence": "Medium",
    "applyReadiness": "Needs visa clarity",
    "breakdown": {
      "registration": 1,
      "visa": 0.5,
      "level": 1,
      "trainingValue": 0.8,
      "practicalFit": 0.9
    },
    "reason": "Short clear reason based only on supplied information.",
    "warning": "Short warning or empty string."
  }
]
`;
}

function cvScoringGenerationInstructions() {
  return `
CV scoring-to-generation workflow:
- Before writing the final CV, internally assess how well the applicant matches the selected job.
- Use the job score, AI score, instant readiness, closing information, and job description if supplied.
- Identify the strongest matching evidence from the applicant's real CV.
- Identify gaps or weak areas that should be handled carefully.
- Improve the CV by strengthening relevant real experience, not by inventing new facts.
- If the job score/readiness suggests visa, AHPRA, supervision, or level concerns, address them honestly in the Registration / Visa / Work Rights section.
- If the job is ED/RMO/HMO focused, push acute care, ED exposure, procedures, escalation, referrals, discharge summaries, and supervised decision-making higher in the CV.
- If the job is rotational or ward-based, push ward care, admissions, documentation, discharge planning, multidisciplinary communication, and follow-up care higher.
- If the job is procedure-heavy, push suturing, I&D, catheterisation, IV/IA sampling, NG tube insertion, wound care, and other real procedures higher.
- Do not print the internal score analysis unless specifically requested.
- The final CV must feel like it was deliberately tailored to improve the applicant's match for this exact job.
`;
}

function makeSeekSlug(query) {
  return query
    .toLowerCase()
    .replace(/\bdoctor\b/g, "")
    .replace(/\bjobs?\b/g, "")
    .replace(/\baustralia\b/g, "")
    .replace(/\bhospital\b/g, "")
    .replace(/[^a-z0-9\s-]/g, "")
    .trim()
    .replace(/\s+/g, "-")
    .replace(/-+/g, "-");
}

async function extractSeekJobsWithPlaywright(url) {
  let browser;

  try {
    browser = await chromium.launch({ headless: true });

    const page = await browser.newPage({
      userAgent:
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/120 Safari/537.36"
    });

    await page.goto(url, { waitUntil: "domcontentloaded", timeout: 30000 });
    await page.waitForTimeout(2500);

    const jobs = await page.evaluate(() => {
      const anchors = Array.from(document.querySelectorAll("a[href*='/job/']"));
      const seen = new Set();
      const results = [];

      for (const a of anchors) {
        const href = a.href;
        const title = (a.innerText || "").trim();

        if (!href || !title || title.length < 4) continue;
        if (seen.has(href)) continue;

        seen.add(href);

        let text = "";
        let parent = a.parentElement;

        for (let i = 0; i < 6 && parent; i++) {
          const parentText = (parent.innerText || "").trim();
          if (parentText.length > text.length) text = parentText;
          parent = parent.parentElement;
        }

        const lines = text.split("\n").map(x => x.trim()).filter(Boolean);

        results.push({
          title,
          employer: "",
          location: "",
          jobType: "",
          link: href,
          snippet: lines.slice(0, 18).join(" | ")
        });
      }

      return results;
    });

    return jobs.slice(0, 100);
  } catch (error) {
    console.error("Playwright SEEK extraction failed:", error.message);
    return [];
  } finally {
    if (browser) await browser.close();
  }
}


function isSeekSearchPage(url) {
  const lower = url.toLowerCase();
  return lower.includes("seek.com.au") && !lower.includes("seek.com.au/job/");
}

function isGenericJobListingPage(item = {}) {
  const url = String(item.url || "").toLowerCase();
  const title = String(item.title || "").toLowerCase();
  const content = String(item.content || item.snippet || "").toLowerCase();
  const combined = `${title} ${content}`;

  if (isSeekSearchPage(url)) return true;

  if (
    combined.match(/^[\d,]+\s+.+\s+jobs\s+in\s+/i) ||
    combined.includes("jobs in australia | jora") ||
    combined.includes("jobs in australia - jora") ||
    combined.includes("jobs in australia | indeed") ||
    combined.includes("jobs in australia - indeed") ||
    combined.includes("hospital medical officer jobs in australia") ||
    combined.includes("resident medical officer jobs in australia") ||
    combined.includes("rmo jobs in australia")
  ) {
    return true;
  }

  const genericTitlePatterns = [
    /jobs in .*\| jora/i,
    /jobs in .*- jora/i,
    /jobs in .*\| indeed/i,
    /jobs in .*- indeed/i,
    /hospital medical officer jobs/i,
    /resident medical officer jobs/i,
    /rmo jobs/i,
    /medical officer jobs/i
  ];

  return genericTitlePatterns.some(pattern => pattern.test(title));
}

function isAllowedUrl(url = "") {
  const value = String(url || "").toLowerCase();

  const allowed = [
    "seek.com.au",
    "smartjobs.qld.gov.au",
    "apply-springboard.health.qld.gov.au",
    "jobs.health.nsw.gov.au",
    "iworkfor.nsw.gov.au",
    "careers.vic.gov.au",
    "health.vic.gov.au",
    "jobs.sa.gov.au",
    "jobs.wa.gov.au",
    "health.nt.gov.au",
    "jobs.act.gov.au",
    "jobs.tas.gov.au",
    "au.jora.com",
    "jora.com",
    "indeed.com",
    "au.indeed.com",
    "linkedin.com/jobs",
    "healthworkforce.com.au",
    "medrecruit.com",
    "skilledmedical.com",
    "headmedical.com",
    "globalmedics.com.au",
    "ramsaycareers.com.au",
    "healthscope.com.au"
  ];

  const blocked = [
    "facebook.com",
    "reddit.com",
    "ahpra.gov.au",
    "medicalboard.gov.au",
    "youtube.com",
    "linkedin.com/in/",
    "linkedin.com/company/",
    "linkedin.com/feed/"
  ];

  return allowed.some(d => value.includes(d)) && !blocked.some(d => value.includes(d));
}

function getJobSource(url = "") {
  const value = String(url || "").toLowerCase();

  if (value.includes("seek.com.au")) return "SEEK";
  if (value.includes("jora.com")) return "Jora";
  if (value.includes("indeed.com")) return "Indeed";
  if (value.includes("linkedin.com")) return "LinkedIn";
  if (value.includes("smartjobs.qld.gov.au") || value.includes("apply-springboard.health.qld.gov.au")) return "Queensland Health / SmartJobs";
  if (value.includes("jobs.health.nsw.gov.au") || value.includes("iworkfor.nsw.gov.au")) return "NSW Health / iWorkForNSW";
  if (value.includes("careers.vic.gov.au") || value.includes("health.vic.gov.au")) return "Victoria Health / Careers VIC";
  if (value.includes("jobs.sa.gov.au")) return "SA Government Jobs";
  if (value.includes("jobs.wa.gov.au")) return "WA Government Jobs";
  if (value.includes("health.nt.gov.au")) return "NT Health";
  if (value.includes("jobs.act.gov.au")) return "ACT Government Jobs";
  if (value.includes("jobs.tas.gov.au")) return "Tasmanian Government Jobs";
  if (value.includes("medrecruit.com")) return "Medrecruit";
  if (value.includes("skilledmedical.com")) return "Skilled Medical";
  if (value.includes("headmedical.com")) return "Head Medical";
  if (value.includes("globalmedics.com.au")) return "Global Medics";
  if (value.includes("ramsaycareers.com.au")) return "Ramsay Careers";
  if (value.includes("healthscope.com.au")) return "Healthscope";
  if (value.includes("healthworkforce.com.au")) return "Health Workforce";

  return "Other job board";
}

function cleanJobTitle(title = "") {
  return String(title || "")
    .replace(/\s*[-|]\s*(SEEK|Jora|Indeed|LinkedIn|SmartJobs|Queensland Government|NSW Health).*$/i, "")
    .replace(/\s*\|\s*.*$/g, "")
    .replace(/\s+/g, " ")
    .trim() || "Untitled job";
}

function extractLocationFromText(text = "") {
  const source = String(text || "");
  const patterns = [
    /(?:location|located in|based in)[:\s-]+([^|\n.]+)/i,
    /\b(Queensland|QLD|New South Wales|NSW|Victoria|VIC|South Australia|SA|Western Australia|WA|Tasmania|TAS|Northern Territory|NT|ACT|Canberra|Sydney|Melbourne|Brisbane|Perth|Adelaide|Hobart|Darwin|Rockhampton|Cairns|Townsville|Mackay|Gold Coast|Sunshine Coast|Toowoomba)\b/i
  ];

  for (const pattern of patterns) {
    const match = source.match(pattern);
    if (match?.[1]) return match[1].trim();
    if (match?.[0]) return match[0].trim();
  }

  return "";
}

function extractEmployerFromText(text = "", source = "") {
  const value = String(text || "");
  const patterns = [
    /(?:company|employer|organisation|organization|facility|hospital|health service)[:\s-]+([^|\n.]+)/i,
    /(?:at|with)\s+([A-Z][A-Za-z&'\s]+(?:Hospital|Health|Health Service|Medical Centre|Clinic|Care|Network))/
  ];

  for (const pattern of patterns) {
    const match = value.match(pattern);
    if (match?.[1]) return match[1].trim();
  }

  if (source && source !== "Other job board") return source;
  return "";
}

function extractJobTypeFromText(text = "") {
  const value = String(text || "").toLowerCase();

  if (value.includes("full time") || value.includes("full-time")) return "Full-time";
  if (value.includes("part time") || value.includes("part-time")) return "Part-time";
  if (value.includes("fixed term") || value.includes("fixed-term")) return "Fixed term";
  if (value.includes("temporary")) return "Temporary";
  if (value.includes("permanent")) return "Permanent";
  if (value.includes("casual")) return "Casual";
  if (value.includes("contract")) return "Contract";

  return "";
}

function buildDirectJobFromTavilyItem(item) {
  const source = getJobSource(item.url || "");
  const rawTitle = item.title || "Untitled job";
  const content = item.content || item.snippet || "";
  const combined = `${rawTitle}\n${content}`;

  return addClosingDateInfo({
    title: cleanJobTitle(rawTitle),
    employer: extractEmployerFromText(combined, source),
    location: extractLocationFromText(combined),
    jobType: extractJobTypeFromText(combined),
    link: item.url,
    snippet: content,
    source,
    descriptionSource: "tavily"
  });
}

function getHardRejectionReason(job) {
  const title = `${job.title || ""}`.toLowerCase();
  const text = `${job.title || ""} ${job.snippet || ""} ${job.employer || ""}`.toLowerCase();

  const blockedTitleTerms = [
    "nurse",
    "nursing",
    "registered nurse",
    "clinical nurse",
    "enrolled nurse",
    "midwife",
    "midwifery",
    "assistant in nursing",
    "support worker",
    "personal care worker",
    "pharmacist",
    "physiotherapist",
    "occupational therapist",
    "psychologist",
    "social worker",
    "dentist",
    "veterinarian",
    "sonographer",
    "radiographer",
    "speech pathologist",
    "dietitian",
    "dental assistant"
  ];

  const tooSeniorForJuniorStream = [
    "staff specialist",
    "consultant physician",
    "consultant psychiatrist",
    "consultant anaesthetist",
    "consultant anesthetist",
    "clinical director",
    "medical director",
    "director of medical services",
    "executive director",
    "specialist consultant"
  ];

  if (blockedTitleTerms.some(term => title.includes(term))) {
    return "Skipped: non-doctor or allied health role.";
  }

  if (tooSeniorForJuniorStream.some(term => text.includes(term))) {
    return "Skipped: consultant/staff specialist/director-level role, not suitable for RMO/PHO stream.";
  }

  if (text.includes("volunteer") || text.includes("unpaid")) {
    return "Skipped: unpaid or volunteer role.";
  }

  if (text.includes("telehealth only") || text.includes("remote only")) {
    return "Skipped: remote/telehealth-only role, likely not suitable for supervised hospital medical work.";
  }

  return "";
}

function isClearlyNonDoctorJob(job) {
  return Boolean(getHardRejectionReason(job));
}

function hasDoctorSignal(job) {
  const text = `${job.title || ""} ${job.snippet || ""}`.toLowerCase();

  const terms = [
    "rmo",
    "resident medical officer",
    "hmo",
    "hospital medical officer",
    "jmo",
    "junior medical officer",
    "pho",
    "principal house officer",
    "medical officer",
    "doctor",
    "registrar",
    "intern",
    "unaccredited registrar",
    "medical practitioner"
  ];

  return terms.some(term => text.includes(term));
}

function getInstantScore(job, userQuery = "") {
  const title = `${job.title || ""}`.toLowerCase();
  const text = `${job.title || ""} ${job.snippet || ""} ${job.location || ""}`.toLowerCase();
  const query = String(userQuery || "").toLowerCase();

  let score = 0;

  if (title.includes("resident medical officer") || title.includes("rmo")) score += 35;
  if (title.includes("hospital medical officer") || title.includes("hmo")) score += 30;
  if (title.includes("junior medical officer") || title.includes("jmo")) score += 28;
  if (title.includes("principal house officer") || title.includes("pho")) score += 32;
  if (title.includes("unaccredited registrar")) score += 24;
  if (title.includes("registrar")) score += 18;
  if (title.includes("medical officer")) score += 22;
  if (title.includes("intern")) score += 18;

  if (text.includes("hospital")) score += 8;
  if (text.includes("health service")) score += 6;
  if (text.includes("queensland health") || text.includes("nsw health") || text.includes("sa health") || text.includes("wa health")) score += 8;

  if (text.includes("emergency") || text.includes("ed ") || text.includes("emergency department")) score += 5;
  if (text.includes("medicine") || text.includes("surgery") || text.includes("icu") || text.includes("critical care")) score += 4;
  if (text.includes("rotation") || text.includes("rotational")) score += 5;
  if (text.includes("supervision") || text.includes("supervised")) score += 8;
  if (text.includes("limited registration") || text.includes("img") || text.includes("international medical graduate")) score += 10;
  if (text.includes("sponsor") || text.includes("sponsorship") || text.includes("482") || text.includes("visa")) score += 10;

  if (query.includes("queensland") && (text.includes("queensland") || text.includes("qld"))) score += 10;
  if (query.includes("new south wales") && (text.includes("new south wales") || text.includes("nsw"))) score += 10;
  if (query.includes("victoria") && (text.includes("victoria") || text.includes("vic"))) score += 10;
  if (query.includes("south australia") && (text.includes("south australia") || text.includes("sa health"))) score += 10;
  if (query.includes("western australia") && (text.includes("western australia") || text.includes("wa health"))) score += 10;

  if (text.includes("casual")) score -= 8;
  if (text.includes("telehealth")) score -= 12;
  if (text.includes("general practitioner") || text.includes("gp only")) score -= 10;
  if (text.includes("senior medical officer") || title.includes("smo")) score -= 12;
  if (text.includes("staff specialist") || text.includes("consultant")) score -= 25;

  return Math.max(0, score);
}

function getInstantReadiness(job) {
  const text = `${job.title || ""} ${job.snippet || ""}`.toLowerCase();
  const hardRejectionReason = getHardRejectionReason(job);

  if (hardRejectionReason) {
    return "Not suitable";
  }

  if (
    text.includes("limited registration") ||
    text.includes("supervision") ||
    text.includes("img") ||
    text.includes("international medical graduate") ||
    text.includes("sponsor") ||
    text.includes("sponsorship") ||
    text.includes("482") ||
    text.includes("visa")
  ) {
    return "High potential";
  }

  if (
    text.includes("rmo") ||
    text.includes("resident medical officer") ||
    text.includes("hmo") ||
    text.includes("hospital medical officer") ||
    text.includes("pho") ||
    text.includes("registrar")
  ) {
    return "Worth reviewing";
  }

  return "Needs manual check";
}

function enrichAndRankJobs(jobs, userQuery = "") {
  return jobs
    .map(job => {
      const instantScore = getInstantScore(job, userQuery);
      const instantReadiness = getInstantReadiness(job);
      const hardRejectionReason = getHardRejectionReason(job);

      return addClosingDateInfo({
        ...job,
        instantScore,
        instantReadiness,
        hardRejectionReason
      });
    })
    .sort((a, b) => b.instantScore - a.instantScore);
}

function normaliseJobKey(job) {
  const title = String(job.title || "")
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .replace(/\s+/g, " ")
    .trim();

  const employer = String(job.employer || "")
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .replace(/\s+/g, " ")
    .trim();

  const location = String(job.location || "")
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .replace(/\s+/g, " ")
    .trim();

  const linkHost = (() => {
    try {
      return new URL(job.link).hostname.replace(/^www\./, "");
    } catch {
      return "";
    }
  })();

  return `${title}|${employer}|${location}|${linkHost}`;
}

function splitAndDedupeJobs(jobs, userQuery = "") {
  const seen = new Set();
  const unique = [];
  const rejected = [];

  for (const job of jobs) {
    if (!job.title || !job.link) continue;

    const hardRejectionReason = getHardRejectionReason(job);
    if (hardRejectionReason) {
      rejected.push({
        ...job,
        numericScore: 0,
        instantScore: 0,
        instantReadiness: "Not suitable",
        hardRejectionReason,
        aiScore: hardRejectionReason
      });
      continue;
    }

    if (!hasDoctorSignal(job)) {
      rejected.push({
        ...job,
        numericScore: 0,
        instantScore: 0,
        instantReadiness: "Not suitable",
        hardRejectionReason: "Skipped: no clear doctor/RMO/HMO/PHO/registrar signal in title or snippet.",
        aiScore: "Skipped: no clear doctor/RMO/HMO/PHO/registrar signal in title or snippet."
      });
      continue;
    }

    const key = normaliseJobKey(job);
    if (seen.has(key)) continue;

    seen.add(key);
    unique.push(job);
  }

  return {
    suitable: enrichAndRankJobs(unique, userQuery),
    rejected: enrichAndRankJobs(rejected, userQuery)
  };
}

function dedupeJobs(jobs, userQuery = "") {
  return splitAndDedupeJobs(jobs, userQuery).suitable;
}

function createDocxFromText(text, filename, res) {
  const lines = text.split("\n");
  const children = [];

  const headingWords = [
    "personal details",
    "professional profile",
    "registration",
    "visa",
    "work rights",
    "key clinical skills",
    "clinical skills",
    "employment history",
    "clinical experience",
    "education",
    "qualifications",
    "courses",
    "certifications",
    "audit",
    "audits",
    "research",
    "publications",
    "teaching",
    "leadership",
    "achievements",
    "referees",
    "cover letter",
    "application email"
  ];

  for (const line of lines) {
    const trimmed = line.trim();

    if (!trimmed) {
      children.push(new Paragraph({ text: "", spacing: { after: 120 } }));
      continue;
    }

    const lower = trimmed.toLowerCase();

    const isHeading =
      headingWords.some(h => lower.includes(h)) &&
      trimmed.length < 90 &&
      !trimmed.startsWith("•") &&
      !trimmed.startsWith("-");

    if (isHeading) {
      children.push(
        new Paragraph({
          children: [
            new TextRun({
              text: trimmed.toUpperCase(),
              bold: true,
              size: 26
            })
          ],
          heading: HeadingLevel.HEADING_2,
          spacing: { before: 300, after: 120 }
        })
      );
      continue;
    }

    if (trimmed.startsWith("•") || trimmed.startsWith("-")) {
      children.push(
        new Paragraph({
          children: [
            new TextRun({
              text: trimmed.replace(/^[-•]\s*/, ""),
              size: 22
            })
          ],
          bullet: { level: 0 },
          spacing: { after: 80 }
        })
      );
      continue;
    }

    const looksLikeName =
      children.length === 0 ||
      (
        children.length < 3 &&
        trimmed.length < 60 &&
        !trimmed.includes(".") &&
        !trimmed.includes(":") &&
        /^[A-Z][A-Za-z\s.'-]+$/.test(trimmed)
      );

    if (looksLikeName && children.length < 3) {
      children.push(
        new Paragraph({
          children: [
            new TextRun({
              text: trimmed,
              bold: true,
              size: 32
            })
          ],
          alignment: AlignmentType.CENTER,
          spacing: { after: 160 }
        })
      );
      continue;
    }

    children.push(
      new Paragraph({
        children: [
          new TextRun({
            text: trimmed,
            size: 22
          })
        ],
        spacing: { after: 100 }
      })
    );
  }

  const doc = new Document({
    sections: [
      {
        properties: {
          page: {
            margin: {
              top: 720,
              right: 720,
              bottom: 720,
              left: 720
            }
          }
        },
        children
      }
    ]
  });

  return Packer.toBuffer(doc).then(buffer => {
    res.setHeader("Content-Disposition", `attachment; filename=${filename}`);
    res.setHeader("Content-Type", "application/vnd.openxmlformats-officedocument.wordprocessingml.document");
    res.send(buffer);
  });
}

app.post("/auto-search", async (req, res) => {
  try {
    const { query, profile } = req.body;

    if (!TAVILY_API_KEY) {
      throw new Error("Missing TAVILY_API_KEY in .env file.");
    }

    const userQuery = query || buildSearchQueryFromProfile(profile);
    logAnalytics("search_started", {
      query: userQuery,
      level: profile?.jobLevel || profile?.level || "",
      state: profile?.stateFilter || "",
      specialty: profile?.specialtyInterest || "",
      needsSponsorship: Boolean(profile?.needsSponsorship)
    });
    console.log("AUTO SEARCH START");
    console.log("Query used:", userQuery);

    const cacheKey = `search:${userQuery.toLowerCase()}`;

    const cached = searchCache.get(cacheKey);
    if (cached) {
      console.log("Returning cached jobs:", cached.length);
      return res.json({ results: cached, cached: true, queryUsed: userQuery });
    }

    const negativeTerms = "-nurse -nursing -midwife -pharmacist -physiotherapist -allied -dentist";
    const sourceQueries = [
      `${userQuery} ${negativeTerms} site:seek.com.au`,
      `${userQuery} ${negativeTerms} site:jora.com OR site:au.jora.com`,
      `${userQuery} ${negativeTerms} site:indeed.com OR site:au.indeed.com`,
      `${userQuery} ${negativeTerms} site:linkedin.com/jobs`,
      `${userQuery} ${negativeTerms} site:smartjobs.qld.gov.au OR site:apply-springboard.health.qld.gov.au`,
      `${userQuery} ${negativeTerms} site:iworkfor.nsw.gov.au OR site:jobs.health.nsw.gov.au`,
      `${userQuery} ${negativeTerms} site:careers.vic.gov.au OR site:health.vic.gov.au`,
      `${userQuery} ${negativeTerms} site:jobs.sa.gov.au OR site:jobs.wa.gov.au`,
      `${userQuery} ${negativeTerms} site:health.nt.gov.au OR site:jobs.act.gov.au OR site:jobs.tas.gov.au`,
      `${userQuery} ${negativeTerms} site:medrecruit.com OR site:skilledmedical.com OR site:headmedical.com OR site:globalmedics.com.au`
    ];

    const tavilyResponses = await Promise.allSettled(
      sourceQueries.map(sourceQuery => tavilyClient.search(sourceQuery, {
        searchDepth: "basic",
        maxResults: 12,
        includeAnswer: false,
        includeRawContent: false,
        topic: "general"
      }))
    );

    const tavilyResults = tavilyResponses.flatMap((response, index) => {
      if (response.status !== "fulfilled") {
        console.warn("Tavily source search failed:", sourceQueries[index], response.reason?.message || response.reason);
        return [];
      }
      return response.value.results || [];
    });

    console.log("Tavily source searches:", sourceQueries.length);
    console.log("Tavily combined raw results:", tavilyResults.length);

    let extractedJobs = [];

    const slug = makeSeekSlug(userQuery) || "rmo";
    const seekBaseUrl = `https://www.seek.com.au/${slug}-jobs`;

    const seekUrls = [
      seekBaseUrl
    ];

    const seekJobGroups = await Promise.all(
      seekUrls.map(url => extractSeekJobsWithPlaywright(url))
    );

    for (const seekJobs of seekJobGroups) {
      extractedJobs.push(...seekJobs);
    }
    console.log("Playwright SEEK direct jobs:", extractedJobs.length);

    const allowedTavilyResults = tavilyResults
      .filter(item => isAllowedUrl(item.url || ""))
      .filter(item => !isGenericJobListingPage(item))
      .slice(0, 80);

    const seekPages = allowedTavilyResults
      .filter(item => isSeekSearchPage(item.url || ""))
      .slice(0, 2);

    const tavilySeekJobGroups = await Promise.all(
      seekPages.map(page => extractSeekJobsWithPlaywright(page.url))
    );

    for (const seekJobs of tavilySeekJobGroups) {
      extractedJobs.push(...seekJobs);
    }
    console.log("After Tavily SEEK page scraping jobs:", extractedJobs.length);

    const directJobs = allowedTavilyResults
      .filter(item => !isSeekSearchPage(item.url || ""))
      .map(buildDirectJobFromTavilyItem);

    extractedJobs.push(...directJobs);
    console.log("Extracted jobs before dedupe:", extractedJobs.length);

    let uniqueJobs = dedupeJobs(extractedJobs, userQuery).slice(0, 80);

    if (uniqueJobs.length === 0) {
      console.log("No jobs after dedupe. Running fallback broad RMO search.");

      const fallbackQuery = "resident medical officer RMO hospital medical officer jobs Australia site:seek.com.au OR site:jora.com OR site:indeed.com OR site:smartjobs.qld.gov.au OR site:jobs.health.nsw.gov.au";
      const fallbackResponse = await tavilyClient.search(fallbackQuery, {
        searchDepth: "basic",
        maxResults: 80,
        includeAnswer: false,
        includeRawContent: false,
        topic: "general"
      });

      const fallbackJobs = (fallbackResponse.results || [])
        .filter(item => isAllowedUrl(item.url || ""))
        .filter(item => !isGenericJobListingPage(item))
        .map(buildDirectJobFromTavilyItem);

      uniqueJobs = dedupeJobs(fallbackJobs, fallbackQuery).slice(0, 80);
      console.log("Fallback unique jobs after dedupe:", uniqueJobs.length);
    }

    console.log("Unique jobs after dedupe:", uniqueJobs.length);
    searchCache.set(cacheKey, uniqueJobs);
    logAnalytics("search_completed", {
      query: userQuery,
      resultCount: uniqueJobs.length
    });
    res.json({ results: uniqueJobs, queryUsed: userQuery });

  } catch (error) {
    logAnalytics("search_error", { message: error.message });
    res.status(500).json({
      error: `Auto-search error: ${error.message}`
    });
  }
});

app.post("/score-jobs", async (req, res) => {
  try {
    const { jobs, quickProfile, profile, offset = 0, limit = 10 } = req.body;
    const applicantProfile = quickProfile || profileToText(profile);

    logAnalytics("score_started", {
      receivedJobs: Array.isArray(jobs) ? jobs.length : 0,
      offset,
      limit
    });

    if (!Array.isArray(jobs) || jobs.length === 0) {
      return res.status(400).json({ error: "No jobs received for scoring." });
    }

    const splitJobs = splitAndDedupeJobs(jobs, applicantProfile);
    const safeOffset = Math.max(0, Number(offset) || 0);
    const safeLimit = Math.min(10, Math.max(1, Number(limit) || 10));
    const jobsSlice = splitJobs.suitable.slice(safeOffset, safeOffset + safeLimit);
    const jobsToScore = await enrichTopJobsWithFirecrawl(jobsSlice);
    const cacheKey = `score:${SCORING_PROVIDER}:${safeOffset}:${safeLimit}:${JSON.stringify(jobsToScore.map(j => j.link))}:${applicantProfile}`;

    const cached = scoreCache.get(cacheKey);
    if (cached) return res.json({ results: cached.results || cached, rejected: cached.rejected || [], cached: true });

    const prompt = `
You are an Australian medical recruitment assistant.

Give PRELIMINARY suitability scoring for each job.
Use only the supplied title, link, employer, location, type, and snippet.
Keep the output compact. Each reason and warning must be under 20 words.
Do not pretend you have read the full official job description.

Applicant:
${applicantProfile || "No profile provided"}

${scoringFramework()}

Jobs:
${jobsToScore.map((job, index) => `
Job ${index + 1}
Title: ${job.title}
Employer: ${job.employer || ""}
Location: ${job.location || ""}
Type: ${job.jobType || ""}
Link: ${job.link}
Snippet: ${(job.snippet || "").slice(0, 500)}
Full job description source: ${job.descriptionSource || "snippet"}
Full job description if available: ${(job.fullDescription || "").slice(0, 1000)}
Closing date: ${job.closingDate || "Not stated"}
Closing status: ${job.closingStatus || "No closing date found"}
Expiry label: ${job.expiryLabel || "Closing date not found"}
Days until closing: ${job.daysUntilClosing ?? "Not available"}
`).join("\n")}
`;

    const text = await askScoringModel(prompt);
    const scores = extractJson(text);

    const scoredJobs = jobsToScore
      .map((job, index) => {
        const score = scores.find(s => Number(s.jobNumber) === index + 1);
        const aiScore = buildScoreText(score);
        const numericScore = Number(score?.score || parseScore(aiScore) || 0);

        return addClosingDateInfo({
          ...job,
          aiScore,
          numericScore,
          scoringProvider: SCORING_PROVIDER,
          descriptionSource: job.descriptionSource || "snippet",
          instantReadiness: job.instantReadiness || getInstantReadiness(job),
          instantScore: job.instantScore || getInstantScore(job),
          hardRejectionReason: job.hardRejectionReason || ""
        });
      })
      .sort((a, b) => b.numericScore - a.numericScore);

    scoreCache.set(cacheKey, {
      results: scoredJobs,
      rejected: splitJobs.rejected,
      nextOffset: safeOffset + scoredJobs.length,
      hasMore: safeOffset + safeLimit < splitJobs.suitable.length,
      totalSuitable: splitJobs.suitable.length
    });

    logAnalytics("score_completed", {
      scoredCount: scoredJobs.length,
      rejectedCount: splitJobs.rejected.length,
      nextOffset: safeOffset + scoredJobs.length,
      totalSuitable: splitJobs.suitable.length,
      provider: SCORING_PROVIDER
    });

    res.json({
      results: scoredJobs,
      rejected: splitJobs.rejected,
      nextOffset: safeOffset + scoredJobs.length,
      hasMore: safeOffset + safeLimit < splitJobs.suitable.length,
      totalSuitable: splitJobs.suitable.length
    });
  } catch (error) {
    logAnalytics("score_error", { message: error.message });
    res.status(500).json({ error: `Score error: ${error.message}` });
  }
});

app.post("/application-pack", async (req, res) => {
  try {
    const { profile, job } = req.body;
    logAnalytics("application_pack_started", {
      jobTitle: extractJobTitleFromJobText(job)
    });

    const prompt = `
You are an Australian medical recruitment assistant.

Prepare a semi-automated application pack for this medical job.
Use a natural, human Australian medical recruitment tone. Avoid robotic phrasing and generic filler.
${mergedCvModeInstructions()}

Do NOT pretend the application has been submitted.
Do NOT answer legal/visa/AHPRA declaration questions automatically.
The applicant must review everything before submitting.

Return:

1. APPLICATION READINESS CHECK
- Apply / Maybe / Skip
- Key reasons
- Registration concerns
- Visa concerns

2. TAILORED CV STRATEGY
- State whether the profile is using pasted CV mode or structured template mode
- Extract the key job criteria in simple bullet points
- What to emphasise for this exact job
- What to de-emphasise or remove
- Which sections should be moved higher
- Missing information to add
- Suggested targeted professional profile paragraph
- Suggested 6 to 10 high-impact CV bullet points matched to the job

3. COVER LETTER
Write a complete professional Australian medical cover letter.

4. SHORT APPLICATION EMAIL
Write a short email to Medical Workforce / recruitment team.

5. SELECTION CRITERIA / STATEMENT RESPONSES
Write short responses using Australian hospital language.

6. QUESTIONS TO ASK HR
Include questions about sponsorship, AHPRA supervision, start date, contract, and referees.

7. FINAL CHECKLIST
Give a practical checklist before applying.

${buildMedicalContext(profile, job)}
`;

    const cacheKey = makeGeminiCacheKey("application-pack", profile, job, MODEL_SMART);
    const result = await askGeminiCached(cacheKey, prompt, MODEL_SMART);
    logAnalytics("application_pack_completed", {
      jobTitle: extractJobTitleFromJobText(job),
      chars: result.length
    });
    res.json({ result });
  } catch (error) {
    logAnalytics("application_pack_error", { message: error.message });
    res.status(500).json({ error: `Application pack error: ${error.message}` });
  }
});

app.post("/application-pack-download", async (req, res) => {
  try {
    const { profile, job } = req.body;
    logAnalytics("application_pack_download_started", {
      jobTitle: extractJobTitleFromJobText(job)
    });

    const prompt = `
Prepare a complete semi-automated Australian medical job application pack.
Use a polished, natural, human Australian medical recruitment tone. Avoid robotic phrasing and generic filler.
${mergedCvModeInstructions()}

Return plain text only.
Include:
- Application readiness check
- Tailored CV strategy, including whether pasted CV mode or structured template mode was used
- Cover letter
- Short application email
- Selection criteria responses
- Questions to ask HR
- Final checklist

Do not say the application has been submitted.

${buildMedicalContext(profile, job)}
`;

    const cacheKey = makeGeminiCacheKey("application-pack-download", profile, job, MODEL_SMART);
    const text = await askGeminiCached(cacheKey, prompt, MODEL_SMART);
    logAnalytics("application_pack_download_completed", {
      jobTitle: extractJobTitleFromJobText(job),
      chars: text.length
    });
    await createDocxFromText(text, buildDownloadFilename(profile, job, "Application_Pack"), res);
  } catch (error) {
    logAnalytics("application_pack_download_error", { message: error.message });
    res.status(500).json({ error: `Application pack download error: ${error.message}` });
  }
});

app.post("/upload-docx", async (req, res) => {
  try {
    if (!req.body.file) return res.status(400).json({ error: "No Word file received." });

    const buffer = Buffer.from(req.body.file, "base64");
    const result = await mammoth.extractRawText({ buffer });
    logAnalytics("cv_uploaded", {
      extractedChars: result.value?.length || 0
    });
    res.json({ text: result.value });
  } catch (error) {
    logAnalytics("cv_upload_error", { message: error.message });
    res.status(500).json({ error: `Word upload error: ${error.message}` });
  }
});

app.post("/evaluate", async (req, res) => {
  try {
    const { profile, job } = req.body;
    logAnalytics("evaluate_started", {
      jobTitle: extractJobTitleFromJobText(job)
    });

    const prompt = `
You are an expert Australian medical recruiter.

This is a FINAL evaluation because the full job description has been provided.
Use a concise, practical Australian medical recruitment style.

${scoringFramework()}

Also include:
- Registration fit explanation
- Visa/sponsorship fit explanation
- Key risks/red flags
- Final recommendation

${buildMedicalContext(profile, job)}
`;

    const cacheKey = makeGeminiCacheKey("evaluate", profile, job, MODEL_SMART);
    const result = await askGeminiCached(cacheKey, prompt, MODEL_SMART);
    logAnalytics("evaluate_completed", {
      jobTitle: extractJobTitleFromJobText(job),
      chars: result.length
    });
    res.json({ result });
  } catch (error) {
    logAnalytics("evaluate_error", { message: error.message });
    res.status(500).json({ error: `Evaluate error: ${error.message}` });
  }
});

app.post("/cv", async (req, res) => {
  try {
    const { profile, job } = req.body;
    logAnalytics("cv_generation_started", {
      jobTitle: extractJobTitleFromJobText(job)
    });

    const prompt = `
You are an expert Australian hospital medical CV writer.

${jobCriteriaExtractionInstructions()}
${cvTailoringInstructions()}
${mergedCvModeInstructions()}
${cvQualityInstructions()}
${humanWritingInstructions()}
${topTierMedicalCvInstructions()}
${cvScoringGenerationInstructions()}
Additional humanisation and de-AI rules:
- After writing the CV, review it and REMOVE any generic or AI-sounding phrases.
- Specifically remove or avoid words like: dedicated, passionate, proven ability, dynamic, highly motivated, strong team player, fast-paced, cutting-edge.
- Replace vague statements with concrete clinical actions.
- Every bullet should reflect a real task or responsibility, not a generic claim.
- Ensure the CV reads like it was written by a real doctor applying for a job, not an AI generator.

Output requirements:
- Return the full tailored CV only.
- Use Australian medical CV headings.
- Use bullet points beginning with "•" for clinical experience, skills, audits, achievements, teaching, and leadership.
- Include a targeted professional profile at the top.
- Include key clinical skills relevant to the job and selected role level.
- Include employment history in reverse chronological order if dates are supplied.
- Include education, registration/visa information if supplied, courses, audits/research/publications, and referees if supplied.
- Keep it honest, polished, and professional.
- Ensure tone is natural, slightly varied, and human (not repetitive or templated).
- Avoid identical sentence structures across bullet points.
- Prefer short, direct clinical statements over long generic sentences.

Smart tailoring rules (very important):
- Identify the top 5–8 most important requirements from the job description.
- Use any supplied job score, AI score, recommendation, apply readiness, warnings, and instant readiness to decide what the CV must strengthen.
- If the score is low or moderate, do not ignore the weakness; improve the CV honestly by highlighting relevant real evidence and using placeholders for missing information.
- Reorder clinical experience so the MOST relevant duties appear first.
- Emphasise:
  • ED experience if job mentions emergency/acute care
  • Procedures if job mentions hands-on skills
  • Teamwork and escalation if hospital-based role
  • Supervision level if limited registration is relevant
  • AMC MCQ, PTE, AHPRA eligibility, and visa/sponsorship status where relevant
- De-emphasise irrelevant experience (do not delete, just move lower).
- Align wording with job keywords without copying the job description.
- Make the CV clearly feel "written for this job".

${buildMedicalContext(profile, job)}
`;

    const cacheKey = makeGeminiCacheKey("cv", profile, job, MODEL_SMART);
    const result = await askGeminiCached(cacheKey, prompt, MODEL_SMART);
    logAnalytics("cv_generation_completed", {
      jobTitle: extractJobTitleFromJobText(job),
      chars: result.length
    });
    res.json({ result });
  } catch (error) {
    logAnalytics("cv_generation_error", { message: error.message });
    res.status(500).json({ error: `CV error: ${error.message}` });
  }
});

app.post("/cv-download", async (req, res) => {
  try {
    const { profile, job, cvText } = req.body;
    logAnalytics("cv_download_started", {
      jobTitle: extractJobTitleFromJobText(job),
      usedExistingText: Boolean(cvText && String(cvText).trim().length > 100)
    });

    if (cvText && String(cvText).trim().length > 100) {
      logAnalytics("cv_download_completed", {
        jobTitle: extractJobTitleFromJobText(job),
        source: "existing_text",
        chars: String(cvText).trim().length
      });
      return await createDocxFromText(String(cvText).trim(), buildDownloadFilename(profile, job, "CV"), res);
    }

    const prompt = `
You are an expert Australian hospital medical CV writer.

Create a top-tier Australian medical CV suitable for Word download using the candidate's selected CV mode.

${jobCriteriaExtractionInstructions()}
${cvTailoringInstructions()}
${mergedCvModeInstructions()}
${cvQualityInstructions()}
${humanWritingInstructions()}
${topTierMedicalCvInstructions()}
${cvScoringGenerationInstructions()}

Rules:
- Return plain text only.
- Do not use markdown symbols.
- Use clear section headings.
- Use bullet points starting with "•" for clinical experience, skills, audits, achievements, teaching, and leadership.
- Return the full CV only.
- Do not fake registration or visa status.
- Tailor strongly to the supplied job description and selected role level.
- Use placeholders for missing facts rather than inventing details.
- Use any supplied job score, AI score, recommendation, apply readiness, warnings, and instant readiness to improve the CV match.
- If the selected job text includes an AI Score section, use it internally to strengthen the CV but do not print the score analysis in the CV.
- Make the downloaded CV use the same smart tailoring and humanisation rules as the preview CV.

${buildMedicalContext(profile, job)}
`;

    const cacheKey = makeGeminiCacheKey("cv-download", profile, job, MODEL_SMART);
    const text = await askGeminiCached(cacheKey, prompt, MODEL_SMART);
    logAnalytics("cv_download_completed", {
      jobTitle: extractJobTitleFromJobText(job),
      source: "generated",
      chars: text.length
    });
    await createDocxFromText(text, buildDownloadFilename(profile, job, "CV"), res);
  } catch (error) {
    logAnalytics("cv_download_error", { message: error.message });
    res.status(500).json({ error: `CV download error: ${error.message}` });
  }
});

app.post("/cv-review", async (req, res) => {
  try {
    const { profile, job, cvText } = req.body;
    logAnalytics("cv_review_started", {
      jobTitle: extractJobTitleFromJobText(job),
      cvChars: String(cvText || "").length
    });

    const prompt = `
You are an Australian medical recruitment reviewer.

Review this CV against the supplied job description.

Return:
1. CV match score out of 5
2. Key strengths
3. Main gaps
4. Missing keywords or criteria
5. Suggested changes to improve match
6. 5 stronger bullet points the applicant could use, without inventing facts
7. What the CV generator should emphasise first when rewriting this CV
8. What should be de-emphasised or moved lower

Rules:
- Be practical and concise.
- Do not invent experience.
- If facts are missing, say what to add as placeholders.
- Focus on Australian hospital medical recruitment.

Doctor Profile:
${profileToText(profile)}

Job Description:
${job || ""}

CV to review:
${cvText || ""}
`;

    const cacheKey = makeGeminiCacheKey("cv-review", profile, `${job || ""}:${cvText || ""}`, MODEL_SMART);
    const result = await askGeminiCached(cacheKey, prompt, MODEL_SMART);
    logAnalytics("cv_review_completed", {
      jobTitle: extractJobTitleFromJobText(job),
      chars: result.length
    });
    res.json({ result });
  } catch (error) {
    logAnalytics("cv_review_error", { message: error.message });
    res.status(500).json({ error: `CV review error: ${error.message}` });
  }
});

app.post("/job-criteria", async (req, res) => {
  try {
    const { profile, job } = req.body;
    logAnalytics("job_criteria_started", {
      jobTitle: extractJobTitleFromJobText(job)
    });

    const prompt = `
You are an Australian medical recruitment assistant.

Extract key criteria from this Australian medical job.

Return:
1. Role level and specialty
2. Must-have criteria
3. Preferred criteria
4. AHPRA/registration clues
5. Visa/sponsorship clues
6. Location and practical considerations
7. Closing date if mentioned
8. Keywords for CV/cover letter
9. Candidate fit notes based on the supplied profile

Rules:
- Be concise and practical.
- Do not invent requirements.
- If unsure, say "Not stated".

${buildMedicalContext(profile, job)}
`;

    const result = await askGeminiCached(
      makeGeminiCacheKey("job-criteria", profile, job, MODEL_SMART),
      prompt,
      MODEL_SMART
    );

    logAnalytics("job_criteria_completed", {
      jobTitle: extractJobTitleFromJobText(job),
      chars: result.length
    });
    res.json({ result });
  } catch (error) {
    logAnalytics("job_criteria_error", { message: error.message });
    res.status(500).json({ error: `Job criteria error: ${error.message}` });
  }
});

app.post("/cv-improve", async (req, res) => {
  try {
    const { profile, job, cvText, reviewText } = req.body;
    logAnalytics("cv_improve_started", {
      jobTitle: extractJobTitleFromJobText(job),
      cvChars: String(cvText || "").length,
      reviewChars: String(reviewText || "").length
    });

    const prompt = `
You are an expert Australian medical CV editor.

Improve this CV for the supplied Australian medical job.

Rules:
- Return the full improved CV only.
- Do not use markdown symbols.
- Do not invent facts, dates, hospitals, qualifications, registration status, visa status, audits, publications, referees, or achievements.
- Use placeholders for missing details, such as [Add date], [Add hospital], or [Add referee details].
- Improve the professional summary, clinical bullet points, section order, and keyword match.
- Use the CV review feedback if supplied.
- If both uploaded/pasted CV content and structured profile fields are present, merge them intelligently: pasted CV is the factual backbone, structured fields fill gaps and add extra details.
- Use the job score/readiness context if present in the job description text.
- Strengthen weak scoring areas honestly by highlighting real matching evidence from the candidate's CV.
- If unsure, prefer omission over guessing.
- Keep the tone natural, human, and suitable for Australian hospital medical recruitment.
- Remove generic or AI-sounding phrases.

Doctor Profile:
${profileToText(profile)}

Job Description:
${job || ""}

CV Review Feedback:
${reviewText || ""}

CV to improve:
${cvText || ""}
`;

    const result = await askGeminiCached(
      makeGeminiCacheKey("cv-improve", profile, `${job || ""}:${cvText || ""}:${reviewText || ""}`, MODEL_SMART),
      prompt,
      MODEL_SMART
    );

    logAnalytics("cv_improve_completed", {
      jobTitle: extractJobTitleFromJobText(job),
      chars: result.length
    });
    res.json({ result });
  } catch (error) {
    logAnalytics("cv_improve_error", { message: error.message });
    res.status(500).json({ error: `CV improve error: ${error.message}` });
  }
});

app.post("/cover-letter", async (req, res) => {
  try {
    const { profile, job } = req.body;
    logAnalytics("cover_letter_started", {
      jobTitle: extractJobTitleFromJobText(job)
    });

    const prompt = `
Write an Australian medical cover letter.
First identify the job's key criteria internally, then match the applicant's real experience to those criteria.
Use a polished, natural, human tone. Avoid generic AI-sounding statements.
${humanWritingInstructions()}

Cover letter style:
- Sound like a real doctor applying for the job, not a template.
- Keep the opening direct and specific to the role.
- Use the applicant's real rotations, clinical duties, AMC result, English test, registration status, and visa/sponsorship situation where relevant.
- Do not overstate Australian experience if the applicant has not worked in Australia.
- Make sponsorship/AHPRA supervision wording honest and professional.
- Keep it warm, clear, and concise.

Important:
- Do not fake registration or visa status.
- Tailor it to the job description.
- If unsure, prefer omission over guessing.

Return:
1. Cover letter
2. Short email text

${buildMedicalContext(profile, job)}
`;

    const cacheKey = makeGeminiCacheKey("cover-letter", profile, job, MODEL_SMART);
    const result = await askGeminiCached(cacheKey, prompt, MODEL_SMART);
    logAnalytics("cover_letter_completed", {
      jobTitle: extractJobTitleFromJobText(job),
      chars: result.length
    });
    res.json({ result });
  } catch (error) {
    logAnalytics("cover_letter_error", { message: error.message });
    res.status(500).json({ error: `Cover letter error: ${error.message}` });
  }
});

app.post("/cover-letter-download", async (req, res) => {
  try {
    const { profile, job, coverLetterText } = req.body;
    logAnalytics("cover_letter_download_started", {
      jobTitle: extractJobTitleFromJobText(job),
      usedExistingText: Boolean(coverLetterText && String(coverLetterText).trim().length > 100)
    });

    if (coverLetterText && String(coverLetterText).trim().length > 100) {
      logAnalytics("cover_letter_download_completed", {
        jobTitle: extractJobTitleFromJobText(job),
        source: "existing_text",
        chars: String(coverLetterText).trim().length
      });
      return await createDocxFromText(String(coverLetterText).trim(), buildDownloadFilename(profile, job, "Cover_Letter"), res);
    }

    const prompt = `
Write an Australian medical cover letter.
First identify the job's key criteria internally, then match the applicant's real experience to those criteria.
Use a polished, natural, human tone. Avoid generic AI-sounding statements.
${humanWritingInstructions()}

Cover letter style:
- Sound like a real doctor applying for the job, not a template.
- Keep the opening direct and specific to the role.
- Use the applicant's real rotations, clinical duties, AMC result, English test, registration status, and visa/sponsorship situation where relevant.
- Do not overstate Australian experience if the applicant has not worked in Australia.
- Make sponsorship/AHPRA supervision wording honest and professional.
- Keep it warm, clear, and concise.

Rules:
- Return plain text only.
- Do not use markdown symbols.
- Do not fake registration or visa status.
- Tailor it to the supplied job description.
- If unsure, prefer omission over guessing.
- Include a short application email after the cover letter.

${buildMedicalContext(profile, job)}
`;

    const cacheKey = makeGeminiCacheKey("cover-letter-download", profile, job, MODEL_SMART);
    const text = await askGeminiCached(cacheKey, prompt, MODEL_SMART);
    logAnalytics("cover_letter_download_completed", {
      jobTitle: extractJobTitleFromJobText(job),
      source: "generated",
      chars: text.length
    });
    await createDocxFromText(text, buildDownloadFilename(profile, job, "Cover_Letter"), res);
  } catch (error) {
    logAnalytics("cover_letter_download_error", { message: error.message });
    res.status(500).json({ error: `Cover letter download error: ${error.message}` });
  }
});

const PORT = process.env.PORT || 3000;

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}. Search cap: 80 jobs. Scoring provider: ${SCORING_PROVIDER}. Firecrawl top jobs: ${FIRECRAWL_TOP_N}. Gemini CV/evaluation model: ${MODEL_SMART}.`);
});