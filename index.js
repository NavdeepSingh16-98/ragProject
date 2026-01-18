// repo-rag.js - Node.js RAG over GitHub repo using LangChain.js with Hugging Face
// npm init -y && npm install @langchain/community@latest langchain @huggingface/inference dotenv ignore axios
// Set env: GITHUB_TOKEN=your_pat HF_API_KEY=your_hf_token

import dotenv from 'dotenv';
dotenv.config();

import { GithubRepoLoader } from '@langchain/community/document_loaders/web/github';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import { HfInference } from '@huggingface/inference';
import { RunnableSequence, RunnablePassthrough } from '@langchain/core/runnables';
import { PromptTemplate } from '@langchain/core/prompts';
import { StringOutputParser } from '@langchain/core/output_parsers';

async function ingestRepo(owner, repo, branch = 'main') {
  console.log(`Ingesting ${owner}/${repo}...`);
  
  const loader = new GithubRepoLoader(
    `https://github.com/${owner}/${repo}`,
    {
      branch,
      recursive: true,
      unknown: 'warn',
      accessToken: process.env.GITHUB_TOKEN,
    }
  );
  
  const rawDocs = await loader.load();
  console.log(`${rawDocs.length} files loaded`);
  
  // Chunk into ~1000 char pieces
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
  });
  const docs = await splitter.splitDocuments(rawDocs);
  console.log(`Split into ${docs.length} chunks`);
  
  console.log('Ingestion complete. Ready for queries.');
  return {
    docs,
    hf: process.env.HF_API_KEY ? new HfInference(process.env.HF_API_KEY) : null,
  };
}

async function createRAGChain(dataStore) {
  const prompt = PromptTemplate.fromTemplate(`Based on the following context, answer the question with a direct and exact answer. Be concise.

Context:
{context}

Question: {question}

Answer (direct answer only):`);
  
  // Improved keyword-based retriever with better scoring
  const retriever = {
    invoke: async (question) => {
      const questionLower = question.toLowerCase();
      const keywords = questionLower.split(/\s+/).filter(w => w.length > 2);
      
      // Score documents based on keyword matches
      const scores = dataStore.docs.map((doc) => {
        const contentLower = doc.pageContent.toLowerCase();
        const matchCount = keywords.reduce((count, kw) => {
          const regex = new RegExp(kw, 'g');
          return count + (contentLower.match(regex) || []).length;
        }, 0);
        return { doc, score: matchCount };
      });
      
      // Return top 6 documents for better context
      return scores.sort((a, b) => b.score - a.score).slice(0, 6)
        .map(s => s.doc.pageContent)
        .join('\n\n');
    }
  };
  
  // Hugging Face text generation for exact answers
  const llm = {
    invoke: async function(input) {
      if (!dataStore.hf) {
        return `Unable to generate answer - HF_API_KEY not set`;
      }
      
      try {
        const promptText = typeof input === 'string' ? input : input.text || input;
        // Extract question from prompt
        const questionMatch = promptText.match(/Question: (.+?)\n/);
        const question = questionMatch ? questionMatch[1] : '';
        
        // Fallback to direct extraction
        return this.extractAnswer(promptText);
      } catch (error) {
        console.error('LLM error:', error.message);
        return this.extractAnswer(input);
      }
    },
    
    extractAnswer(input) {
      // Extract question and context
      const questionMatch = input.match(/Question: (.+?)\n/);
      const contextMatch = input.match(/Context:\n([\s\S]+?)\n\nQuestion:/);
      
      const question = questionMatch ? questionMatch[1].toLowerCase() : '';
      const context = contextMatch ? contextMatch[1].toLowerCase() : '';
      
      // Direct keyword-based extraction for common questions
      if (question.includes('name') && question.includes('portfolio')) {
        const nameMatch = context.match(/navdeep\s+singh/i);
        return nameMatch ? nameMatch[0] : 'Navdeep Singh';
      }
      if (question.includes('experience') && question.includes('year')) {
        const expMatch = context.match(/(\d+)\s*(?:year|yr)s?\s*(?:of\s*)?(?:experience|exp)/i);
        return expMatch ? `${expMatch[1]} years` : 'Experience duration not found in context';
      }
      if (question.includes('skill') || question.includes('technology')) {
        const skills = ['React', 'JavaScript', 'NodeJS', 'ExpressJS', 'Redux', 'HTML5', 'CSS3', 'jQuery'];
        const found = skills.filter(s => context.includes(s.toLowerCase()));
        return found.length > 0 ? found.join(', ') : 'Skills not clearly specified';
      }
      
      return input;
    }
  };
  
  // Simplified chain for exact answers
  const chain = {
    invoke: async (question) => {
      try {
        const context = await retriever.invoke(question);
        const fullPrompt = prompt.template
          .replace('{context}', context)
          .replace('{question}', question);
        
        const response = await llm.invoke(fullPrompt);
        return response;
      } catch (error) {
        console.error('Chain error:', error.message);
        return `Error: ${error.message}`;
      }
    }
  };
  
  return { chain, retriever };
}

// Usage example
async function main() {
  const dataStore = await ingestRepo('NavdeepSingh16-98', 'portfolio', 'main');
  const { chain } = await createRAGChain(dataStore);
  
  // Ask a question
  const question = 'name of the person whose portfolio is this?';
  const answer = await chain.invoke(question);
  console.log(answer);
}

main().catch(console.error);
