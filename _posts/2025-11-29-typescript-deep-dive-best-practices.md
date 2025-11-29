---
layout: post
title: "🌊 TypeScript Deep Dive & Best Practices (2025)"
description: "A comprehensive guide to TypeScript covering fundamental concepts, advanced patterns, and industry best practices for 2025"
author: technical_notes
date: 2025-11-29 00:00:00 +0530
categories: [Notes, TypeScript]
tags: [Programming, TypeScript, JavaScript, Type System, Best Practices, Web Development]
image: /assets/img/posts/typescript-logo.png
toc: true
math: true
mermaid: true
---

## Table of Contents

1. [Introduction to TypeScript](#introduction-to-typescript)
2. [Jargon Tables](#jargon-tables)
3. [Core Type System](#core-type-system)
4. [Type Annotations & Inference](#type-annotations--inference)
5. [Advanced Type Patterns](#advanced-type-patterns)
6. [Generics](#generics)
7. [Utility Types](#utility-types)
8. [Type Narrowing & Guards](#type-narrowing--guards)
9. [Conditional Types](#conditional-types)
10. [Mapped Types](#mapped-types)
11. [Template Literal Types](#template-literal-types)
12. [Configuration & Compiler](#configuration--compiler)
13. [Best Practices](#best-practices)
14. [Common Pitfalls](#common-pitfalls)
15. [References](#references)

---

## Introduction to TypeScript

TypeScript is a statically-typed superset of JavaScript that compiles to plain JavaScript. Developed and maintained by Microsoft, it adds optional static types, classes, and modules to JavaScript, enabling developers to catch errors at compile-time rather than runtime.

### Why TypeScript?

- **Type Safety**: Catch errors during development before code runs
- **Enhanced IDE Support**: Intelligent code completion, refactoring, and navigation
- **Better Documentation**: Types serve as inline documentation
- **Scalability**: Easier to maintain large codebases
- **Modern JavaScript Features**: Use latest ECMAScript features with backward compatibility

### Current Status (2025)

The stable TypeScript ecosystem in 2025 includes:
- **Latest Stable Version**: TypeScript 5.x series
- **Primary Models**: Claude Sonnet 4.5 (claude-sonnet-4-5-20250929) and Claude Haiku 4.5 (claude-haiku-4-5-20251001)
- **Key Features**: Decorators (Stage 3), improved type inference, performance optimizations

---

## Jargon Tables

### Table 1: TypeScript Lifecycle Terminology

This table maps common jargon used across different contexts for similar concepts in the TypeScript development lifecycle.

| Concept | TypeScript Term | JavaScript Equivalent | Alternative Names | Description |
|---------|----------------|----------------------|------------------|-------------|
| **Source Processing** | Transpilation | Compilation | Transformation, Trans-compilation | Converting TypeScript to JavaScript |
| **Source Processing** | Compilation | - | Type Checking, Build | Full process including type checking |
| **Type Resolution** | Type Inference | - | Type Deduction, Implicit Typing | Automatic type determination |
| **Type Resolution** | Type Annotation | Type Declaration | Explicit Typing, Type Signature | Manual type specification |
| **Type Refinement** | Type Narrowing | Type Guards | Type Refinement, Flow Analysis | Making types more specific |
| **Type Checking** | Static Analysis | - | Compile-time Checking | Analyzing code before runtime |
| **Type System** | Structural Typing | Duck Typing | Shape Matching | Type compatibility based on structure |
| **Code Generation** | Emit | Output, Compilation Output | Code Generation | Producing JavaScript files |
| **Type Definition** | Declaration Files (.d.ts) | Type Definitions | Ambient Declarations | Type information for JavaScript libraries |
| **Type Safety** | Strict Mode | - | Strict Type Checking | Enabling all strict type-checking options |

### Table 2: Hierarchical Differentiation of TypeScript Jargon

This table organizes TypeScript terminology hierarchically from broad concepts to specific implementations.

| Level | Category | Subcategory | Specific Terms | Relationship |
|-------|----------|-------------|----------------|--------------|
| **1** | **Type System** | - | - | Root concept |
| **2** | | Primitive Types | `string`, `number`, `boolean`, `null`, `undefined`, `symbol`, `bigint` | Built-in basic types |
| **2** | | Object Types | `object`, `array`, `tuple`, `enum` | Composite types |
| **2** | | Special Types | `any`, `unknown`, `never`, `void` | Utility types for specific scenarios |
| **3** | | | `any` - Escape hatch | Disables type checking |
| **3** | | | `unknown` - Type-safe any | Requires type checking |
| **3** | | | `never` - Impossible values | Represents values that never occur |
| **3** | | | `void` - No return value | Function returns nothing |
| **1** | **Type Operations** | - | - | Ways to work with types |
| **2** | | Type Creation | Union, Intersection, Literal Types | Building new types |
| **3** | | | Union (`\|`) | Type can be one of several types |
| **3** | | | Intersection (`&`) | Type must satisfy all types |
| **3** | | | Literal | Exact values as types |
| **2** | | Type Transformation | Mapped Types, Conditional Types | Deriving types from existing ones |
| **3** | | | Mapped Types | Transform properties of types |
| **3** | | | Conditional Types | Type logic with conditions |
| **2** | | Type Refinement | Narrowing, Guards, Predicates | Making types more specific |
| **3** | | | Type Guards | Runtime checks for types |
| **3** | | | Type Predicates | Custom type guard functions |
| **3** | | | Control Flow Analysis | Automatic narrowing |
| **1** | **Generics** | - | - | Parameterized types |
| **2** | | Generic Functions | Type Parameters | Reusable function types |
| **2** | | Generic Classes | Type Parameters | Reusable class types |
| **2** | | Generic Constraints | `extends` keyword | Limiting generic types |
| **1** | **Advanced Patterns** | - | - | Complex type constructs |
| **2** | | Utility Types | Built-in helpers | Standard type transformations |
| **3** | | | `Partial<T>`, `Required<T>`, `Readonly<T>` | Property modifiers |
| **3** | | | `Pick<T, K>`, `Omit<T, K>` | Property selection |
| **3** | | | `Record<K, T>`, `Extract<T, U>`, `Exclude<T, U>` | Type construction |
| **2** | | Template Literal Types | String manipulation | Type-level string operations |
| **2** | | Index Signatures | Dynamic property types | Flexible object shapes |
| **1** | **Configuration** | - | - | Compiler settings |
| **2** | | Compiler Options | tsconfig.json | Project configuration |
| **2** | | Strict Options | Type checking modes | Safety levels |
| **3** | | | `strict`, `noImplicitAny`, `strictNullChecks` | Individual strict options |
| **2** | | Module Resolution | Import/Export strategies | Code organization |

---

## Core Type System

### Primitive Types

TypeScript supports all JavaScript primitive types with static type annotations.

```typescript
// String
let userName: string = "Alice";

// Number (integers and floats)
let age: number = 30;
let price: number = 99.99;

// Boolean
let isActive: boolean = true;

// BigInt (for large integers)
let bigNumber: bigint = 9007199254740991n;

// Symbol (unique identifiers)
let uniqueId: symbol = Symbol("id");

// Null and Undefined
let empty: null = null;
let notDefined: undefined = undefined;
```

### Special Types

```typescript
// any - Disables type checking (use sparingly)
let flexible: any = 4;
flexible = "now I'm a string";
flexible = false; // No error

// unknown - Type-safe alternative to any
let uncertain: unknown = 4;
// uncertain.toFixed(); // Error: must check type first
if (typeof uncertain === "number") {
  uncertain.toFixed(2); // OK after type guard
}

// never - Represents values that never occur
function throwError(message: string): never {
  throw new Error(message);
}

// void - Function returns nothing
function logMessage(message: string): void {
  console.log(message);
  // No return statement needed
}
```

### Array Types

```typescript
// Array syntax variations
let numbers: number[] = [1, 2, 3, 4, 5];
let strings: Array<string> = ["a", "b", "c"];

// Readonly arrays
let immutableNumbers: readonly number[] = [1, 2, 3];
// immutableNumbers.push(4); // Error: push does not exist

// Mixed arrays using union types
let mixed: (number | string)[] = [1, "two", 3, "four"];
```

### Tuple Types

Tuples are arrays with fixed length and known types at each position.

```typescript
// Basic tuple
let coordinate: [number, number] = [10, 20];

// Tuple with optional elements
let optionalTuple: [string, number?] = ["hello"];

// Tuple with rest elements
let restTuple: [string, ...number[]] = ["items", 1, 2, 3, 4];

// Named tuples (TypeScript 4.0+)
type Range = [start: number, end: number];
let range: Range = [0, 100];

// Destructuring tuples
let [x, y] = coordinate;
```

### Object Types

```typescript
// Object type annotation
let person: { name: string; age: number } = {
  name: "Bob",
  age: 25
};

// Optional properties
let config: { host: string; port?: number } = {
  host: "localhost"
};

// Readonly properties
let point: { readonly x: number; readonly y: number } = {
  x: 10,
  y: 20
};
// point.x = 15; // Error: cannot reassign readonly property

// Index signatures for dynamic properties
let dictionary: { [key: string]: string } = {
  hello: "world",
  foo: "bar"
};
```

### Enum Types

Enums allow defining named constants.

```typescript
// Numeric enum (default)
enum Direction {
  Up,    // 0
  Down,  // 1
  Left,  // 2
  Right  // 3
}

// String enum
enum Color {
  Red = "RED",
  Green = "GREEN",
  Blue = "BLUE"
}

// Heterogeneous enum (not recommended)
enum Mixed {
  No = 0,
  Yes = "YES"
}

// Const enum (inlined at compile time)
const enum HttpStatus {
  OK = 200,
  NotFound = 404,
  ServerError = 500
}

// Using enums
let direction: Direction = Direction.Up;
let color: Color = Color.Red;
```

### Union Types

Union types allow a value to be one of several types.

```typescript
// Basic union
let id: number | string;
id = 123;      // OK
id = "ABC123"; // OK

// Union with null/undefined
let nullableString: string | null | undefined;

// Union of object types
type Success = { status: "success"; data: string };
type Error = { status: "error"; message: string };
type Result = Success | Error;

// Discriminated unions (tagged unions)
function handleResult(result: Result) {
  if (result.status === "success") {
    console.log(result.data); // Type narrowed to Success
  } else {
    console.log(result.message); // Type narrowed to Error
  }
}
```

### Intersection Types

Intersection types combine multiple types into one.

```typescript
// Basic intersection
type Person = { name: string };
type Employee = { employeeId: number };
type Worker = Person & Employee;

let worker: Worker = {
  name: "Alice",
  employeeId: 12345
};

// Intersecting with utility types
type RequiredPerson = Required<Person> & { age: number };

// Function intersections
type Loggable = { log: () => void };
type Serializable = { serialize: () => string };
type LoggableSerializable = Loggable & Serializable;
```

### Literal Types

Literal types represent exact values.

```typescript
// String literals
let httpMethod: "GET" | "POST" | "PUT" | "DELETE";
httpMethod = "GET"; // OK
// httpMethod = "PATCH"; // Error

// Numeric literals
let diceRoll: 1 | 2 | 3 | 4 | 5 | 6;

// Boolean literals (less common)
let truthyValue: true;

// Template literal types
type Greeting = `Hello, ${string}`;
let greeting: Greeting = "Hello, World"; // OK
```

---

## Type Annotations & Inference

### Type Annotations

Type annotations explicitly specify the type of a variable, parameter, or return value.

```typescript
// Variable annotations
let explicitString: string = "Hello";
let explicitNumber: number = 42;

// Function parameter and return type annotations
function add(a: number, b: number): number {
  return a + b;
}

// Arrow function annotations
const multiply = (a: number, b: number): number => a * b;

// Object destructuring with annotations
function printCoordinate({ x, y }: { x: number; y: number }): void {
  console.log(`X: ${x}, Y: ${y}`);
}
```

### Type Inference

TypeScript automatically infers types when they're not explicitly specified.

```typescript
// Variable inference
let inferredString = "Hello"; // inferred as string
let inferredNumber = 42;      // inferred as number

// Best common type inference
let mixedArray = [0, 1, null]; // inferred as (number | null)[]

// Contextual typing (from context)
window.addEventListener("click", (event) => {
  // event is inferred as MouseEvent
  console.log(event.clientX, event.clientY);
});

// Return type inference
function getUser() {
  return { name: "Alice", age: 30 };
} // Return type inferred as { name: string; age: number; }

// Inference with generics
let numberArray = [1, 2, 3]; // inferred as number[]
let firstElement = numberArray[0]; // inferred as number
```

### When to Use Annotations vs Inference

**Use explicit annotations when:**
- Declaring variables without initialization
- Function parameters (always required)
- Function return types (recommended for public APIs)
- Complex types that benefit from documentation
- You want to enforce a specific type

**Rely on inference when:**
- Initializing variables with values
- Simple return types
- Local variables in small scopes
- The inferred type is obvious

```typescript
// Good: explicit annotation needed
let userId: string | number;

// Good: inference is clear
let userName = "Alice";

// Good: explicit return type for public API
export function calculateTotal(items: number[]): number {
  return items.reduce((sum, item) => sum + item, 0);
}

// Good: inference for internal helper
function internal Helper() {
  return { x: 10, y: 20 };
}
```

---

## Advanced Type Patterns

### Type Aliases

Type aliases create custom names for types.

```typescript
// Basic type alias
type ID = string | number;

// Object type alias
type User = {
  id: ID;
  name: string;
  email: string;
};

// Function type alias
type MathOperation = (a: number, b: number) => number;

const add: MathOperation = (a, b) => a + b;
const subtract: MathOperation = (a, b) => a - b;

// Union type alias
type Status = "pending" | "approved" | "rejected";

// Recursive type alias
type JSONValue =
  | string
  | number
  | boolean
  | null
  | JSONValue[]
  | { [key: string]: JSONValue };
```

### Interfaces

Interfaces define contracts for object shapes.

```typescript
// Basic interface
interface Person {
  name: string;
  age: number;
}

// Optional properties
interface Config {
  host: string;
  port?: number;
  timeout?: number;
}

// Readonly properties
interface Point {
  readonly x: number;
  readonly y: number;
}

// Function properties
interface Calculator {
  add(a: number, b: number): number;
  subtract(a: number, b: number): number;
}

// Extending interfaces
interface Employee extends Person {
  employeeId: number;
  department: string;
}

// Multiple inheritance
interface Timestamped {
  createdAt: Date;
  updatedAt: Date;
}

interface Document extends Person, Timestamped {
  title: string;
  content: string;
}
```

### Type Aliases vs Interfaces

Key differences and when to use each:

```typescript
// Interface: can be extended and merged
interface Animal {
  name: string;
}

interface Animal {
  age: number; // Declaration merging
}

// Type alias: cannot be merged, more flexible
type Pet = {
  name: string;
} & {
  age: number;
};

// Type alias: can represent primitives, unions, tuples
type StringOrNumber = string | number;
type Coordinate = [number, number];

// Interface: preferred for object shapes
interface User {
  id: number;
  name: string;
}

// Interface: better for class contracts
class UserImpl implements User {
  constructor(public id: number, public name: string) {}
}
```

**Guidelines:**
- Use **interfaces** for object shapes and class contracts
- Use **type aliases** for unions, tuples, primitives, and complex types
- Prefer **interfaces** for public APIs (extensible)
- Use **type aliases** when you need unions or mapped types

### Assertion Functions

Assertion functions narrow types by throwing errors when conditions fail.

```typescript
// Basic assertion function
function assertIsString(value: unknown): asserts value is string {
  if (typeof value !== "string") {
    throw new Error("Value must be a string");
  }
}

// Using assertion function
function processValue(value: unknown) {
  assertIsString(value);
  // After this point, value is narrowed to string
  console.log(value.toUpperCase());
}

// Assertion with type predicate
function assertIsNonNull<T>(value: T): asserts value is NonNullable<T> {
  if (value === null || value === undefined) {
    throw new Error("Value is null or undefined");
  }
}

// Using in code
let maybeString: string | null = getValue();
assertIsNonNull(maybeString);
console.log(maybeString.length); // No error, type narrowed
```

---

## Generics

Generics provide a way to create reusable components that work with multiple types.

### Basic Generic Functions

```typescript
// Generic function
function identity<T>(value: T): T {
  return value;
}

// Usage with explicit type
let result1 = identity<string>("hello");

// Usage with type inference
let result2 = identity(42); // T inferred as number

// Generic function with multiple type parameters
function pair<T, U>(first: T, second: U): [T, U] {
  return [first, second];
}

let stringNumberPair = pair("age", 30);
let booleanStringPair = pair<boolean, string>(true, "active");
```

### Generic Constraints

Constraints limit the types that can be used with generics.

```typescript
// Constraint using extends
function getProperty<T, K extends keyof T>(obj: T, key: K): T[K] {
  return obj[key];
}

let person = { name: "Alice", age: 30 };
let name = getProperty(person, "name"); // OK
// let invalid = getProperty(person, "address"); // Error

// Constraint to ensure method exists
interface HasLength {
  length: number;
}

function logLength<T extends HasLength>(item: T): void {
  console.log(item.length);
}

logLength("hello"); // OK, string has length
logLength([1, 2, 3]); // OK, array has length
// logLength(42); // Error, number doesn't have length

// Multiple constraints
function merge<T extends object, U extends object>(obj1: T, obj2: U): T & U {
  return { ...obj1, ...obj2 };
}
```

### Generic Classes

```typescript
// Generic class
class Box<T> {
  private value: T;

  constructor(value: T) {
    this.value = value;
  }

  getValue(): T {
    return this.value;
  }

  setValue(value: T): void {
    this.value = value;
  }
}

let numberBox = new Box<number>(42);
let stringBox = new Box("hello");

// Generic class with constraints
class DataStore<T extends { id: number }> {
  private items: T[] = [];

  add(item: T): void {
    this.items.push(item);
  }

  getById(id: number): T | undefined {
    return this.items.find(item => item.id === id);
  }
}

// Generic class with static members
class Container<T> {
  private value: T;
  static defaultValue: number = 0; // Static members can't use class type parameters

  constructor(value: T) {
    this.value = value;
  }
}
```

### Generic Interfaces

```typescript
// Generic interface
interface Repository<T> {
  getAll(): T[];
  getById(id: number): T | undefined;
  add(item: T): void;
  delete(id: number): boolean;
}

// Implementation
class UserRepository implements Repository<User> {
  private users: User[] = [];

  getAll(): User[] {
    return this.users;
  }

  getById(id: number): User | undefined {
    return this.users.find(user => user.id === id);
  }

  add(user: User): void {
    this.users.push(user);
  }

  delete(id: number): boolean {
    const index = this.users.findIndex(user => user.id === id);
    if (index !== -1) {
      this.users.splice(index, 1);
      return true;
    }
    return false;
  }
}
```

### Default Type Parameters

```typescript
// Generic with default type
interface Result<T = string> {
  data: T;
  success: boolean;
}

let stringResult: Result = { data: "hello", success: true };
let numberResult: Result<number> = { data: 42, success: true };

// Function with default generic
function createArray<T = number>(length: number, value: T): T[] {
  return Array(length).fill(value);
}

let numbers = createArray(5, 0); // T defaults to number
let strings = createArray<string>(3, "hello");
```

---

## Utility Types

TypeScript provides built-in utility types for common type transformations.

### Property Modifiers

```typescript
// Partial<T> - Makes all properties optional
interface User {
  id: number;
  name: string;
  email: string;
}

type PartialUser = Partial<User>;
// Equivalent to:
// {
//   id?: number;
//   name?: string;
//   email?: string;
// }

function updateUser(user: User, updates: Partial<User>): User {
  return { ...user, ...updates };
}

// Required<T> - Makes all properties required
type RequiredConfig = Required<{
  host?: string;
  port?: number;
}>;
// Equivalent to: { host: string; port: number; }

// Readonly<T> - Makes all properties readonly
type ReadonlyUser = Readonly<User>;
const user: ReadonlyUser = { id: 1, name: "Alice", email: "alice@example.com" };
// user.name = "Bob"; // Error: cannot reassign readonly property
```

### Property Selection

```typescript
// Pick<T, K> - Creates a type with selected properties
type UserPreview = Pick<User, "id" | "name">;
// Equivalent to: { id: number; name: string; }

// Omit<T, K> - Creates a type without specified properties
type UserWithoutEmail = Omit<User, "email">;
// Equivalent to: { id: number; name: string; }

// Real-world example
interface Product {
  id: number;
  name: string;
  description: string;
  price: number;
  inStock: boolean;
}

type ProductSummary = Pick<Product, "id" | "name" | "price">;
type ProductForm = Omit<Product, "id">;
```

### Type Extraction

```typescript
// Extract<T, U> - Extracts types from union that are assignable to U
type Mixed = string | number | boolean;
type StringsAndNumbers = Extract<Mixed, string | number>;
// Result: string | number

// Exclude<T, U> - Removes types from union that are assignable to U
type NonBoolean = Exclude<Mixed, boolean>;
// Result: string | number

// NonNullable<T> - Removes null and undefined
type MaybeString = string | null | undefined;
type DefiniteString = NonNullable<MaybeString>;
// Result: string

// Practical example
type PrimitiveTypes = string | number | boolean | null | undefined;
type ActualPrimitives = NonNullable<PrimitiveTypes>;
// Result: string | number | boolean
```

### Function Utilities

```typescript
// ReturnType<T> - Extracts return type of function
function getUser() {
  return { id: 1, name: "Alice", email: "alice@example.com" };
}

type User = ReturnType<typeof getUser>;
// Result: { id: number; name: string; email: string; }

// Parameters<T> - Extracts parameter types as tuple
function createUser(name: string, age: number, active: boolean) {
  return { name, age, active };
}

type CreateUserParams = Parameters<typeof createUser>;
// Result: [name: string, age: number, active: boolean]

// ConstructorParameters<T> - Extracts constructor parameter types
class Person {
  constructor(public name: string, public age: number) {}
}

type PersonConstructorParams = ConstructorParameters<typeof Person>;
// Result: [name: string, age: number]

// InstanceType<T> - Extracts instance type of class
type PersonInstance = InstanceType<typeof Person>;
// Result: Person
```

### Record and Other Utilities

```typescript
// Record<K, T> - Creates object type with keys K and values T
type UserRoles = Record<string, string>;
const roles: UserRoles = {
  admin: "Administrator",
  user: "Regular User",
  guest: "Guest User"
};

// More specific keys
type PageInfo = Record<"home" | "about" | "contact", { title: string; url: string }>;

// Awaited<T> - Unwraps Promise type
type AsyncUser = Promise<User>;
type SyncUser = Awaited<AsyncUser>;
// Result: User

// Practical example with multiple Promise levels
type DeepPromise = Promise<Promise<string>>;
type Unwrapped = Awaited<DeepPromise>;
// Result: string
```

---

## Type Narrowing & Guards

Type narrowing is the process of refining a broader type to a more specific type based on runtime checks.

### typeof Type Guards

```typescript
function processValue(value: string | number) {
  if (typeof value === "string") {
    // value is narrowed to string
    console.log(value.toUpperCase());
  } else {
    // value is narrowed to number
    console.log(value.toFixed(2));
  }
}

// Multiple typeof checks
function handleInput(input: string | number | boolean) {
  if (typeof input === "string") {
    return input.trim();
  } else if (typeof input === "number") {
    return input * 2;
  } else {
    return !input;
  }
}
```

### instanceof Type Guards

```typescript
class Dog {
  bark() {
    console.log("Woof!");
  }
}

class Cat {
  meow() {
    console.log("Meow!");
  }
}

function makeSound(animal: Dog | Cat) {
  if (animal instanceof Dog) {
    animal.bark(); // animal narrowed to Dog
  } else {
    animal.meow(); // animal narrowed to Cat
  }
}

// With error handling
function processError(error: Error | string) {
  if (error instanceof Error) {
    console.log(error.message);
    console.log(error.stack);
  } else {
    console.log(error);
  }
}
```

### in Operator Narrowing

```typescript
interface Bird {
  fly(): void;
  layEggs(): void;
}

interface Fish {
  swim(): void;
  layEggs(): void;
}

function move(animal: Bird | Fish) {
  if ("fly" in animal) {
    animal.fly(); // animal narrowed to Bird
  } else {
    animal.swim(); // animal narrowed to Fish
  }
}

// Checking multiple properties
function processShape(shape: { radius: number } | { width: number; height: number }) {
  if ("radius" in shape) {
    console.log("Circle area:", Math.PI * shape.radius ** 2);
  } else {
    console.log("Rectangle area:", shape.width * shape.height);
  }
}
```

### Equality Narrowing

```typescript
function handleValue(value: string | number | boolean, flag: string | boolean) {
  if (value === flag) {
    // Both narrowed to string | boolean (excluding number)
    console.log(value, flag);
  }
}

// Null/undefined checks
function printLength(text: string | null | undefined) {
  if (text !== null && text !== undefined) {
    console.log(text.length); // text narrowed to string
  }
  
  // Alternative: truthiness check
  if (text) {
    console.log(text.length); // text narrowed to string
  }
}
```

### Truthiness Narrowing

```typescript
function processString(str: string | null | undefined) {
  // Truthiness check narrows away null and undefined
  if (str) {
    console.log(str.toUpperCase());
  }
}

// Be careful with falsy values
function multiplyValue(value: number | undefined, multiplier: number) {
  // This would fail for value = 0
  // if (value) { ... }
  
  // Correct approach
  if (value !== undefined) {
    return value * multiplier;
  }
  return 0;
}

// Array truthiness
function sumArray(numbers: number[] | undefined) {
  if (numbers && numbers.length > 0) {
    return numbers.reduce((sum, n) => sum + n, 0);
  }
  return 0;
}
```

### User-Defined Type Guards

Type predicates allow you to create custom type guard functions.

```typescript
// Basic type predicate
function isString(value: unknown): value is string {
  return typeof value === "string";
}

function processValue(value: unknown) {
  if (isString(value)) {
    console.log(value.toUpperCase()); // value narrowed to string
  }
}

// Complex type guards for objects
interface User {
  name: string;
  email: string;
}

function isUser(obj: unknown): obj is User {
  return (
    typeof obj === "object" &&
    obj !== null &&
    "name" in obj &&
    "email" in obj &&
    typeof (obj as User).name === "string" &&
    typeof (obj as User).email === "string"
  );
}

// Using type guard with API data
async function fetchUser(id: number): Promise<User> {
  const response = await fetch(`/api/users/${id}`);
  const data = await response.json();
  
  if (!isUser(data)) {
    throw new Error("Invalid user data");
  }
  
  return data; // data narrowed to User
}

// Generic type guard
function isArrayOf<T>(
  value: unknown,
  checker: (item: unknown) => item is T
): value is T[] {
  return Array.isArray(value) && value.every(checker);
}

// Usage
const maybeNumbers: unknown = [1, 2, 3];
if (isArrayOf(maybeNumbers, (item): item is number => typeof item === "number")) {
  // maybeNumbers is now number[]
  const sum = maybeNumbers.reduce((a, b) => a + b, 0);
}
```

### Discriminated Unions

Discriminated unions (tagged unions) use a common property to narrow types.

```typescript
// Define discriminated union
interface Circle {
  kind: "circle";
  radius: number;
}

interface Square {
  kind: "square";
  sideLength: number;
}

interface Rectangle {
  kind: "rectangle";
  width: number;
  height: number;
}

type Shape = Circle | Square | Rectangle;

// Narrowing with discriminant property
function getArea(shape: Shape): number {
  switch (shape.kind) {
    case "circle":
      // shape narrowed to Circle
      return Math.PI * shape.radius ** 2;
    case "square":
      // shape narrowed to Square
      return shape.sideLength ** 2;
    case "rectangle":
      // shape narrowed to Rectangle
      return shape.width * shape.height;
  }
}

// Exhaustiveness checking
function assertNever(value: never): never {
  throw new Error(`Unexpected value: ${value}`);
}

function getPerimeter(shape: Shape): number {
  switch (shape.kind) {
    case "circle":
      return 2 * Math.PI * shape.radius;
    case "square":
      return 4 * shape.sideLength;
    case "rectangle":
      return 2 * (shape.width + shape.height);
    default:
      return assertNever(shape); // Ensures all cases handled
  }
}

// Real-world example: API responses
type ApiSuccess<T> = {
  status: "success";
  data: T;
};

type ApiError = {
  status: "error";
  message: string;
  code: number;
};

type ApiResponse<T> = ApiSuccess<T> | ApiError;

function handleResponse<T>(response: ApiResponse<T>): T {
  if (response.status === "success") {
    return response.data; // narrowed to ApiSuccess<T>
  } else {
    throw new Error(`API Error ${response.code}: ${response.message}`);
  }
}
```

---

## Conditional Types

Conditional types create types based on conditions, similar to ternary operators.

### Basic Conditional Types

```typescript
// Syntax: T extends U ? X : Y
type IsString<T> = T extends string ? true : false;

type A = IsString<string>;  // true
type B = IsString<number>;  // false

// Practical example: filtering null/undefined
type NonNullable<T> = T extends null | undefined ? never : T;

type C = NonNullable<string | null>;      // string
type D = NonNullable<number | undefined>; // number

// Extracting function return type
type ReturnType<T> = T extends (...args: any[]) => infer R ? R : never;

type E = ReturnType<() => string>;          // string
type F = ReturnType<(x: number) => number>; // number
```

### Distributive Conditional Types

Conditional types distribute over union types.

```typescript
// Automatically distributes over unions
type ToArray<T> = T extends any ? T[] : never;

type G = ToArray<string | number>;
// Evaluates to: string[] | number[]
// Not: (string | number)[]

// Non-distributive version (using tuple)
type ToArrayNonDist<T> = [T] extends [any] ? T[] : never;

type H = ToArrayNonDist<string | number>;
// Evaluates to: (string | number)[]

// Practical use: filtering types
type ExtractStrings<T> = T extends string ? T : never;

type I = ExtractStrings<"a" | "b" | 1 | 2>;
// Result: "a" | "b"
```

### Infer Keyword

The `infer` keyword extracts types from within conditional types.

```typescript
// Extract array element type
type ArrayElement<T> = T extends (infer E)[] ? E : never;

type J = ArrayElement<number[]>;     // number
type K = ArrayElement<string[]>;     // string

// Extract promise type
type UnwrapPromise<T> = T extends Promise<infer U> ? U : T;

type L = UnwrapPromise<Promise<string>>; // string
type M = UnwrapPromise<number>;          // number

// Extract function parameters
type FirstParameter<T> = T extends (first: infer F, ...args: any[]) => any ? F : never;

type N = FirstParameter<(name: string, age: number) => void>; // string

// Extract object property type
type PropertyType<T, K extends keyof T> = T extends { [P in K]: infer V } ? V : never;

interface User {
  id: number;
  name: string;
}

type O = PropertyType<User, "name">; // string

// Nested infer for complex structures
type DeepArrayElement<T> = T extends (infer E)[]
  ? E extends (infer F)[]
    ? F
    : E
  : never;

type P = DeepArrayElement<number[][]>; // number
```

### Advanced Conditional Patterns

```typescript
// Recursive conditional types
type Flatten<T> = T extends any[]
  ? T[number] extends any[]
    ? Flatten<T[number]>
    : T[number]
  : T;

type Q = Flatten<number[][][]>; // number

// Conditional type with multiple constraints
type SmartExtract<T, U> = T extends U
  ? T
  : T extends any[]
    ? SmartExtract<T[number], U>[]
    : never;

// Type-level if-else chains
type TypeName<T> = T extends string
  ? "string"
  : T extends number
    ? "number"
    : T extends boolean
      ? "boolean"
      : T extends undefined
        ? "undefined"
        : T extends Function
          ? "function"
          : "object";

type R = TypeName<string>;      // "string"
type S = TypeName<() => void>;  // "function"

// Real-world example: form field types
type FieldType<T> = T extends string
  ? HTMLInputElement
  : T extends number
    ? HTMLInputElement
    : T extends boolean
      ? HTMLInputElement
      : T extends Date
        ? HTMLInputElement
        : HTMLTextAreaElement;
```

---

## Mapped Types

Mapped types transform properties of existing types.

### Basic Mapped Types

```typescript
// Syntax: { [P in K]: T }
type ReadonlyType<T> = {
  readonly [P in keyof T]: T[P];
};

interface User {
  id: number;
  name: string;
}

type ReadonlyUser = ReadonlyType<User>;
// Result: { readonly id: number; readonly name: string; }

// Optional mapped type
type OptionalType<T> = {
  [P in keyof T]?: T[P];
};

type PartialUser = OptionalType<User>;
// Result: { id?: number; name?: string; }

// Nullable mapped type
type Nullable<T> = {
  [P in keyof T]: T[P] | null;
};
```

### Mapping Modifiers

```typescript
// Adding modifiers with + (default)
type AddReadonly<T> = {
  +readonly [P in keyof T]: T[P];
};

// Removing modifiers with -
type RemoveReadonly<T> = {
  -readonly [P in keyof T]: T[P];
};

type RemoveOptional<T> = {
  [P in keyof T]-?: T[P];
};

// Combining modifiers
type Mutable<T> = {
  -readonly [P in keyof T]: T[P];
};

type Required<T> = {
  [P in keyof T]-?: T[P];
};

// Real-world example
interface Config {
  readonly host: string;
  readonly port?: number;
  readonly ssl?: boolean;
}

type MutableConfig = Mutable<Config>;
// Result: { host: string; port?: number; ssl?: boolean; }

type CompleteConfig = Required<Config>;
// Result: { readonly host: string; readonly port: number; readonly ssl: boolean; }
```

### Key Remapping

```typescript
// Remapping keys with 'as' clause
type Getters<T> = {
  [P in keyof T as `get${Capitalize<string & P>}`]: () => T[P];
};

interface Person {
  name: string;
  age: number;
}

type PersonGetters = Getters<Person>;
// Result:
// {
//   getName: () => string;
//   getAge: () => number;
// }

// Filtering properties
type RemoveKindField<T> = {
  [P in keyof T as Exclude<P, "kind">]: T[P];
};

interface Product {
  kind: string;
  name: string;
  price: number;
}

type ProductWithoutKind = RemoveKindField<Product>;
// Result: { name: string; price: number; }

// Conditional key remapping
type OnlyStrings<T> = {
  [P in keyof T as T[P] extends string ? P : never]: T[P];
};

interface Mixed {
  id: number;
  name: string;
  email: string;
  active: boolean;
}

type StringFields = OnlyStrings<Mixed>;
// Result: { name: string; email: string; }
```

### Advanced Mapped Types

```typescript
// Mapping with template literals
type EventHandlers<T> = {
  [P in keyof T as `on${Capitalize<string & P>}Change`]: (value: T[P]) => void;
};

interface FormData {
  username: string;
  email: string;
  age: number;
}

type FormHandlers = EventHandlers<FormData>;
// Result:
// {
//   onUsernameChange: (value: string) => void;
//   onEmailChange: (value: string) => void;
//   onAgeChange: (value: number) => void;
// }

// Deep mapped types
type DeepReadonly<T> = {
  readonly [P in keyof T]: T[P] extends object ? DeepReadonly<T[P]> : T[P];
};

interface NestedData {
  user: {
    profile: {
      name: string;
      age: number;
    };
  };
}

type ImmutableData = DeepReadonly<NestedData>;
// All properties at all levels become readonly

// Mapped type with type transformation
type Promisify<T> = {
  [P in keyof T]: Promise<T[P]>;
};

interface SyncAPI {
  getUser: User;
  getPost: Post;
}

type AsyncAPI = Promisify<SyncAPI>;
// Result:
// {
//   getUser: Promise<User>;
//   getPost: Promise<Post>;
// }
```

---

## Template Literal Types

Template literal types build new string literal types using string interpolation.

### Basic Template Literals

```typescript
// Basic template literal type
type World = "world";
type Greeting = `hello ${World}`;
// Result: "hello world"

// Union in template literals
type Color = "red" | "blue" | "green";
type Quantity = "one" | "two";
type ColoredBall = `${Quantity} ${Color} ball`;
// Result: "one red ball" | "one blue ball" | "one green ball" |
//         "two red ball" | "two blue ball" | "two green ball"

// Multiple unions (cartesian product)
type Size = "small" | "medium" | "large";
type Style = "casual" | "formal";
type Outfit = `${Size}-${Style}`;
// Result: "small-casual" | "small-formal" | "medium-casual" | etc.
```

### Intrinsic String Manipulation Types

```typescript
// Uppercase<S> - Converts to uppercase
type UpperGreeting = Uppercase<"hello">;
// Result: "HELLO"

// Lowercase<S> - Converts to lowercase
type LowerGreeting = Lowercase<"HELLO">;
// Result: "hello"

// Capitalize<S> - Capitalizes first character
type CapitalGreeting = Capitalize<"hello">;
// Result: "Hello"

// Uncapitalize<S> - Uncapitalizes first character
type UncapitalGreeting = Uncapitalize<"Hello">;
// Result: "hello"

// Combining with template literals
type HTTPMethod = "get" | "post" | "put" | "delete";
type MethodName = `handle${Capitalize<HTTPMethod>}`;
// Result: "handleGet" | "handlePost" | "handlePut" | "handleDelete"

type EventName = "click" | "focus" | "blur";
type EventHandler = `on${Capitalize<EventName>}`;
// Result: "onClick" | "onFocus" | "onBlur"
```

### Practical Applications

```typescript
// API endpoint types
type Entity = "user" | "post" | "comment";
type Action = "create" | "read" | "update" | "delete";
type Endpoint = `/${Entity}/${Action}`;
// Result: "/user/create" | "/user/read" | "/post/create" | etc.

// CSS property types
type CSSProp = "color" | "background" | "border";
type CSSValue = "red" | "blue" | "1px solid black";
type CSSRule = `${CSSProp}: ${CSSValue}`;

// Database table names
type TablePrefix = "tbl" | "view";
type TableName = "users" | "posts" | "comments";
type FullTableName = `${TablePrefix}_${TableName}`;
// Result: "tbl_users" | "tbl_posts" | "view_users" | etc.

// Type-safe event emitter
type EventMap = {
  login: { username: string };
  logout: void;
  error: { message: string; code: number };
};

type EventNames = keyof EventMap;
type EventHandler<K extends EventNames> = EventMap[K] extends void
  ? () => void
  : (data: EventMap[K]) => void;

class TypedEventEmitter {
  on<K extends EventNames>(event: K, handler: EventHandler<K>): void {
    // Implementation
  }
  
  emit<K extends EventNames>(
    event: K,
    ...args: EventMap[K] extends void ? [] : [EventMap[K]]
  ): void {
    // Implementation
  }
}

// Usage
const emitter = new TypedEventEmitter();
emitter.on("login", (data) => console.log(data.username)); // Typed!
emitter.on("logout", () => console.log("Logged out"));
emitter.emit("error", { message: "Failed", code: 500 });
```

### String Pattern Matching

```typescript
// Extract parts from string patterns
type ExtractRouteParams<T extends string> = T extends `${infer _Start}:${infer Param}/${infer Rest}`
  ? Param | ExtractRouteParams<`/${Rest}`>
  : T extends `${infer _Start}:${infer Param}`
    ? Param
    : never;

type Route1 = ExtractRouteParams<"/users/:id/posts/:postId">;
// Result: "id" | "postId"

type Route2 = ExtractRouteParams<"/products/:productId">;
// Result: "productId"

// Type-safe router
type RouteParams<T extends string> = {
  [K in ExtractRouteParams<T>]: string;
};

function navigate<T extends string>(
  route: T,
  params: RouteParams<T>
): void {
  // Implementation
}

// Usage
navigate("/users/:id/posts/:postId", { id: "123", postId: "456" }); // OK
// navigate("/users/:id", { postId: "456" }); // Error: missing 'id'
```

---

## Configuration & Compiler

### tsconfig.json Basics

The `tsconfig.json` file configures TypeScript compiler options.

```json
{
  "compilerOptions": {
    // Target JavaScript version
    "target": "ES2022",
    
    // Module system
    "module": "ESNext",
    
    // Module resolution strategy
    "moduleResolution": "bundler",
    
    // Output directory
    "outDir": "./dist",
    
    // Root directory of source files
    "rootDir": "./src",
    
    // Enable all strict type-checking options
    "strict": true,
    
    // Generate source maps for debugging
    "sourceMap": true,
    
    // Generate declaration files (.d.ts)
    "declaration": true,
    
    // Allow importing .json files
    "resolveJsonModule": true,
    
    // Ensure consistent casing in imports
    "forceConsistentCasingInFileNames": true,
    
    // Skip type checking of declaration files
    "skipLibCheck": true,
    
    // Interop between CommonJS and ES modules
    "esModuleInterop": true,
    
    // Allow default imports from modules with no default export
    "allowSyntheticDefaultImports": true,
    
    // Enable experimental decorators
    "experimentalDecorators": true,
    
    // Include type definitions
    "lib": ["ES2022", "DOM", "DOM.Iterable"]
  },
  
  // Files to include
  "include": ["src/**/*"],
  
  // Files to exclude
  "exclude": ["node_modules", "dist", "**/*.spec.ts"]
}
```

### Strict Mode Options

Strict mode enables all strict type-checking options. Understanding each helps fine-tune type safety.

```json
{
  "compilerOptions": {
    // Enable all strict options (recommended)
    "strict": true,
    
    // Individual strict options (when strict: true, all are enabled):
    
    // No implicit 'any' type
    "noImplicitAny": true,
    
    // Strict null checks
    "strictNullChecks": true,
    
    // Strict function types
    "strictFunctionTypes": true,
    
    // Strict 'bind', 'call', 'apply'
    "strictBindCallApply": true,
    
    // Strict property initialization
    "strictPropertyInitialization": true,
    
    // No implicit 'this'
    "noImplicitThis": true,
    
    // Always emit 'use strict'
    "alwaysStrict": true
  }
}
```

### Examples of Strict Options

```typescript
// noImplicitAny
// Error: Parameter 'x' implicitly has an 'any' type
function logValue(x) { 
  console.log(x);
}

// Fix: Add explicit type
function logValue(x: unknown) {
  console.log(x);
}

// strictNullChecks
let name: string = "Alice";
// name = null; // Error: Type 'null' is not assignable to type 'string'

// Fix: Use union type
let nullableName: string | null = "Alice";
nullableName = null; // OK

// strictPropertyInitialization
class User {
  // Error: Property 'name' has no initializer
  name: string;
  
  // Fix 1: Initialize in declaration
  email: string = "";
  
  // Fix 2: Initialize in constructor
  constructor(name: string) {
    this.name = name;
  }
  
  // Fix 3: Definite assignment assertion (use sparingly)
  id!: number;
}

// noImplicitThis
const obj = {
  value: 42,
  getValue() {
    // Error: 'this' implicitly has type 'any'
    return function() {
      return this.value;
    };
  }
};

// Fix: Use arrow function
const obj2 = {
  value: 42,
  getValue() {
    return () => this.value; // OK
  }
};
```

### Additional Compiler Options

```json
{
  "compilerOptions": {
    // Linting options
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noImplicitReturns": true,
    "noFallthroughCasesInSwitch": true,
    "noUncheckedIndexedAccess": true,
    
    // Advanced options
    "allowUnreachableCode": false,
    "allowUnusedLabels": false,
    "exactOptionalPropertyTypes": true,
    "noImplicitOverride": true,
    "noPropertyAccessFromIndexSignature": true,
    
    // Module options
    "baseUrl": "./",
    "paths": {
      "@/*": ["src/*"],
      "@components/*": ["src/components/*"],
      "@utils/*": ["src/utils/*"]
    },
    
    // Emit options
    "removeComments": true,
    "importHelpers": true,
    "downlevelIteration": true,
    
    // Debugging
    "inlineSourceMap": false,
    "inlineSources": false
  }
}
```

### Project References

For monorepos or large projects, use project references.

```json
// Root tsconfig.json
{
  "files": [],
  "references": [
    { "path": "./packages/core" },
    { "path": "./packages/utils" },
    { "path": "./packages/ui" }
  ]
}

// packages/core/tsconfig.json
{
  "compilerOptions": {
    "composite": true,
    "declaration": true,
    "declarationMap": true,
    "outDir": "./dist",
    "rootDir": "./src"
  },
  "include": ["src/**/*"]
}

// packages/ui/tsconfig.json
{
  "compilerOptions": {
    "composite": true,
    "outDir": "./dist",
    "rootDir": "./src"
  },
  "references": [
    { "path": "../core" },
    { "path": "../utils" }
  ],
  "include": ["src/**/*"]
}
```

---

## Best Practices

### Type Annotation Guidelines

```typescript
// ✅ Good: Let inference work for simple cases
const count = 42;
const message = "Hello";
const isActive = true;

// ❌ Bad: Unnecessary annotations
const count: number = 42;
const message: string = "Hello";

// ✅ Good: Annotate function parameters
function greet(name: string): string {
  return `Hello, ${name}`;
}

// ✅ Good: Annotate when type can't be inferred
let userId: string | number;
userId = 123;
userId = "ABC";

// ✅ Good: Annotate complex return types for clarity
function fetchUserData(): Promise<{ id: number; name: string }> {
  return fetch("/api/user").then(res => res.json());
}

// ✅ Good: Annotate public APIs
export function calculateTotal(items: Item[]): number {
  return items.reduce((sum, item) => sum + item.price, 0);
}
```

### Prefer Interfaces for Object Shapes

```typescript
// ✅ Good: Interface for object shapes
interface User {
  id: number;
  name: string;
  email: string;
}

// ✅ Good: Type for unions and complex types
type Status = "pending" | "approved" | "rejected";
type Result<T> = { success: true; data: T } | { success: false; error: string };

// ✅ Good: Interface can be extended and merged
interface Animal {
  name: string;
}

interface Animal {
  age: number; // Declaration merging
}

// ✅ Good: Extending interfaces
interface Dog extends Animal {
  breed: string;
}
```

### Use Unknown Instead of Any

```typescript
// ❌ Bad: Using any defeats type safety
function processValue(value: any) {
  return value.toUpperCase(); // No error, but could crash
}

// ✅ Good: Using unknown requires type checking
function processValueSafe(value: unknown) {
  if (typeof value === "string") {
    return value.toUpperCase(); // Safe
  }
  throw new Error("Expected string");
}

// ✅ Good: Type guard with unknown
function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}
```

### Avoid Non-Null Assertions

```typescript
// ❌ Bad: Non-null assertion bypasses safety
function getUser(id: number) {
  return users.find(u => u.id === id)!; // Might be undefined!
}

// ✅ Good: Handle null/undefined explicitly
function getUserSafe(id: number): User | undefined {
  return users.find(u => u.id === id);
}

// ✅ Good: Throw error if null is invalid
function getUserOrThrow(id: number): User {
  const user = users.find(u => u.id === id);
  if (!user) {
    throw new Error(`User ${id} not found`);
  }
  return user;
}

// ⚠️ Acceptable: When you're certain (e.g., DOM after check)
if (document.getElementById("app")) {
  const app = document.getElementById("app")!;
  app.innerHTML = "Hello";
}
```

### Prefer Type Guards Over Type Assertions

```typescript
// ❌ Bad: Type assertion without validation
function handleData(data: unknown) {
  const user = data as User;
  console.log(user.name); // Unsafe
}

// ✅ Good: Type guard with validation
function handleDataSafe(data: unknown) {
  if (isUser(data)) {
    console.log(data.name); // Safe
  }
}

function isUser(value: unknown): value is User {
  return (
    typeof value === "object" &&
    value !== null &&
    "name" in value &&
    "email" in value
  );
}
```

### Use Const Assertions

```typescript
// Without const assertion
const colors = ["red", "green", "blue"];
// Type: string[]

// ✅ Good: With const assertion
const colorsConst = ["red", "green", "blue"] as const;
// Type: readonly ["red", "green", "blue"]

// ✅ Good: Object const assertion
const config = {
  apiUrl: "https://api.example.com",
  timeout: 5000,
} as const;
// All properties become readonly

// ✅ Good: Enum alternative with const assertion
const HttpStatus = {
  OK: 200,
  NotFound: 404,
  ServerError: 500,
} as const;

type HttpStatusCode = typeof HttpStatus[keyof typeof HttpStatus];
// Type: 200 | 404 | 500
```

### Discriminated Unions for State Management

```typescript
// ✅ Good: Discriminated union for loading states
type AsyncData<T> =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "success"; data: T }
  | { status: "error"; error: Error };

function handleAsyncData<T>(state: AsyncData<T>) {
  switch (state.status) {
    case "idle":
      return "Not started";
    case "loading":
      return "Loading...";
    case "success":
      return state.data; // Type-safe access
    case "error":
      return state.error.message;
  }
}

// ✅ Good: Form validation state
type FormState =
  | { kind: "editing"; draft: string }
  | { kind: "validating"; draft: string }
  | { kind: "valid"; value: string }
  | { kind: "invalid"; draft: string; errors: string[] };
```

### Generic Constraints

```typescript
// ❌ Bad: Overly permissive generic
function getValue<T>(obj: T, key: string) {
  return obj[key]; // Error: key not constrained
}

// ✅ Good: Constrain generic appropriately
function getValueSafe<T, K extends keyof T>(obj: T, key: K): T[K] {
  return obj[key];
}

// ✅ Good: Multiple constraints
function merge<T extends object, U extends object>(a: T, b: U): T & U {
  return { ...a, ...b };
}

// ✅ Good: Constraint with interface
interface Identifiable {
  id: number;
}

function findById<T extends Identifiable>(items: T[], id: number): T | undefined {
  return items.find(item => item.id === id);
}
```

### Utility Types for Transformations

```typescript
// ✅ Good: Use built-in utility types
interface User {
  id: number;
  name: string;
  email: string;
  password: string;
}

// Public user (omit sensitive fields)
type PublicUser = Omit<User, "password">;

// User update (all fields optional)
type UserUpdate = Partial<User>;

// Required fields only
type UserRequired = Required<Pick<User, "name" | "email">>;

// ✅ Good: Combine utility types
type CreateUserDto = Omit<User, "id"> & { confirmPassword: string };
```

### Type-Safe Event Handling

```typescript
// ✅ Good: Type-safe event handlers
type EventMap = {
  click: MouseEvent;
  input: InputEvent;
  change: Event;
};

class TypedElement {
  addEventListener<K extends keyof EventMap>(
    type: K,
    listener: (event: EventMap[K]) => void
  ): void {
    // Implementation
  }
}

// Usage
const element = new TypedElement();
element.addEventListener("click", (event) => {
  console.log(event.clientX); // Typed as MouseEvent
});
```

### Organize Types

```typescript
// ✅ Good: Group related types
// types/user.ts
export interface User {
  id: number;
  name: string;
  email: string;
}

export type UserId = User["id"];

export type UserRole = "admin" | "user" | "guest";

export interface UserWithRole extends User {
  role: UserRole;
}

// types/api.ts
export interface ApiResponse<T> {
  data: T;
  status: number;
  message?: string;
}

export type ApiError = {
  error: string;
  code: number;
};

// ✅ Good: Use index file for exports
// types/index.ts
export * from "./user";
export * from "./api";
export * from "./common";
```

### Avoid Magic Numbers and Strings

```typescript
// ❌ Bad: Magic values
function checkStatus(status: number) {
  if (status === 200) {
    return "OK";
  }
}

// ✅ Good: Named constants
const HttpStatus = {
  OK: 200,
  NotFound: 404,
  ServerError: 500,
} as const;

function checkStatusSafe(status: number) {
  if (status === HttpStatus.OK) {
    return "OK";
  }
}

// ✅ Good: Literal types
type HttpMethod = "GET" | "POST" | "PUT" | "DELETE";

function makeRequest(method: HttpMethod, url: string) {
  // Implementation
}
```

---

## Common Pitfalls

### Optional Chaining Misuse

```typescript
// ⚠️ Pitfall: Optional chaining returns undefined
const user: User | undefined = getUser();
const name = user?.name; // Type: string | undefined
// const upperName = name.toUpperCase(); // Error!

// ✅ Solution: Check or provide default
const upperName = user?.name?.toUpperCase() ?? "UNKNOWN";

// ✅ Solution: Type guard
if (user?.name) {
  const upperName = user.name.toUpperCase();
}
```

### Array.prototype.find Type Issues

```typescript
// ⚠️ Pitfall: find returns T | undefined
const users: User[] = getUsers();
const user = users.find(u => u.id === 1);
// user.name; // Error: Object is possibly 'undefined'

// ✅ Solution: Check before use
if (user) {
  console.log(user.name);
}

// ✅ Solution: Provide fallback
const userName = users.find(u => u.id === 1)?.name ?? "Unknown";

// ✅ Solution: Throw if not found
function findUserOrThrow(id: number): User {
  const user = users.find(u => u.id === id);
  if (!user) {
    throw new Error(`User ${id} not found`);
  }
  return user;
}
```

### Index Signature Issues

```typescript
// ⚠️ Pitfall: Index signatures allow any string key
interface Config {
  host: string;
  port: number;
  [key: string]: string | number;
}

const config: Config = {
  host: "localhost",
  port: 3000,
  database: "mydb", // OK but no autocomplete
};

// ✅ Solution: Use Record for dynamic keys
interface KnownConfig {
  host: string;
  port: number;
}

type Config = KnownConfig & Record<string, unknown>;

// ✅ Better: Use Map for truly dynamic data
const settings = new Map<string, string | number>();
settings.set("host", "localhost");
settings.set("port", 3000);
```

### Enum Pitfalls

```typescript
// ⚠️ Pitfall: Numeric enums have reverse mapping
enum Color {
  Red,
  Green,
  Blue,
}

console.log(Color[0]); // "Red" - can cause confusion

// ✅ Solution: Use string enums
enum ColorString {
  Red = "RED",
  Green = "GREEN",
  Blue = "BLUE",
}

// ✅ Better: Use const object with const assertion
const ColorConst = {
  Red: "RED",
  Green: "GREEN",
  Blue: "BLUE",
} as const;

type Color = typeof ColorConst[keyof typeof ColorConst];
```

### Function Overload Complexity

```typescript
// ⚠️ Pitfall: Complex overloads are hard to maintain
function processValue(value: string): string;
function processValue(value: number): number;
function processValue(value: boolean): boolean;
function processValue(value: string | number | boolean): string | number | boolean {
  return value;
}

// ✅ Solution: Use generic or union when possible
function processValueGeneric<T extends string | number | boolean>(value: T): T {
  return value;
}

// ✅ Solution: Use discriminated unions
type Input =
  | { type: "string"; value: string }
  | { type: "number"; value: number }
  | { type: "boolean"; value: boolean };

function processInput(input: Input) {
  switch (input.type) {
    case "string":
      return input.value.toUpperCase();
    case "number":
      return input.value * 2;
    case "boolean":
      return !input.value;
  }
}
```

### Type vs Interface Confusion

```typescript
// ⚠️ Pitfall: Mixing types and interfaces inconsistently
type UserType = {
  name: string;
};

interface UserInterface {
  name: string;
}

// ✅ Solution: Be consistent
// For object shapes: prefer interfaces
interface User {
  id: number;
  name: string;
}

interface Admin extends User {
  permissions: string[];
}

// For unions, intersections: use types
type Status = "active" | "inactive" | "pending";
type Result<T> = Success<T> | Failure;
```

### Generic Constraint Errors

```typescript
// ⚠️ Pitfall: Not constraining generics properly
function getProperty<T>(obj: T, key: string) {
  return obj[key]; // Error: Type 'string' can't be used to index type 'T'
}

// ✅ Solution: Proper constraint
function getPropertySafe<T, K extends keyof T>(obj: T, key: K): T[K] {
  return obj[key];
}

// ⚠️ Pitfall: Over-constraining
function compareValues<T extends number>(a: T, b: T): boolean {
  return a === b; // Too restrictive
}

// ✅ Solution: Minimal constraint
function compareValuesSafe<T>(a: T, b: T): boolean {
  return a === b;
}
```

### Async/Await Type Issues

```typescript
// ⚠️ Pitfall: Not handling Promise rejection types
async function fetchData(): Promise<Data> {
  try {
    const response = await fetch("/api/data");
    return response.json();
  } catch (error) {
    // error is type 'unknown'
    console.log(error.message); // Error!
  }
}

// ✅ Solution: Type guard for errors
async function fetchDataSafe(): Promise<Data> {
  try {
    const response = await fetch("/api/data");
    return response.json();
  } catch (error) {
    if (error instanceof Error) {
      console.log(error.message);
    }
    throw error;
  }
}

// ✅ Solution: Custom error handling
function isApiError(error: unknown): error is ApiError {
  return (
    typeof error === "object" &&
    error !== null &&
    "code" in error &&
    "message" in error
  );
}

async function fetchDataWithCustomError(): Promise<Data> {
  try {
    const response = await fetch("/api/data");
    return response.json();
  } catch (error) {
    if (isApiError(error)) {
      console.log(`API Error ${error.code}: ${error.message}`);
    }
    throw error;
  }
}
```

### Circular Dependencies

```typescript
// ⚠️ Pitfall: Circular type dependencies
// user.ts
import { Post } from "./post";

export interface User {
  id: number;
  name: string;
  posts: Post[];
}

// post.ts
import { User } from "./user";

export interface Post {
  id: number;
  title: string;
  author: User;
}

// ✅ Solution: Use type imports
// user.ts
import type { Post } from "./post";

export interface User {
  id: number;
  name: string;
  posts: Post[];
}

// ✅ Solution: Separate shared types
// types.ts
export interface User {
  id: number;
  name: string;
  postIds: number[];
}

export interface Post {
  id: number;
  title: string;
  authorId: number;
}
```

---

## References

<div class="references-section" markdown="1">

1. <a href="https://www.typescriptlang.org/docs/" target="_blank">TypeScript Official Documentation</a>
2. <a href="https://www.typescriptlang.org/docs/handbook/intro.html" target="_blank">TypeScript Handbook</a>
3. <a href="https://www.typescriptlang.org/tsconfig" target="_blank">TSConfig Reference</a>
4. <a href="https://www.typescriptlang.org/docs/handbook/utility-types.html" target="_blank">TypeScript Utility Types</a>
5. <a href="https://www.typescriptlang.org/docs/handbook/2/narrowing.html" target="_blank">TypeScript Type Narrowing</a>
6. <a href="https://www.typescriptlang.org/docs/handbook/2/generics.html" target="_blank">TypeScript Generics</a>
7. <a href="https://www.typescriptlang.org/docs/handbook/2/conditional-types.html" target="_blank">TypeScript Conditional Types</a>
8. <a href="https://www.typescriptlang.org/docs/handbook/2/mapped-types.html" target="_blank">TypeScript Mapped Types</a>
9. <a href="https://www.typescriptlang.org/docs/handbook/2/template-literal-types.html" target="_blank">TypeScript Template Literal Types</a>
10. <a href="https://github.com/microsoft/TypeScript/wiki/Performance" target="_blank">TypeScript Performance Wiki</a>

</div>

---

## Summary

TypeScript enhances JavaScript with static typing, providing safety, maintainability, and developer experience improvements. Key takeaways:

- **Type System**: Leverages structural typing with primitives, objects, unions, intersections, and literals
- **Inference**: Balances explicit annotations with powerful type inference
- **Advanced Types**: Generics, conditional types, mapped types, and template literals enable sophisticated type transformations
- **Utility Types**: Built-in helpers streamline common type operations
- **Configuration**: Strict mode and compiler options enforce type safety
- **Best Practices**: Prefer interfaces for objects, unknown over any, type guards over assertions, and discriminated unions for state

Mastering TypeScript requires understanding both its type system mechanics and practical patterns. Start with strict mode enabled, leverage type inference where appropriate, and gradually adopt advanced patterns as needs arise.
